import Foundation
import XCTest
@testable import Siliconed

/// **The peak without the kernel's reset** (`FootprintInterval`): the symbol forced absent,
/// the 10 ms sampler must catch an allocation touched then freed between two `reset`s — what the
/// footprint at the boundary alone would miss — and stop with the render.
final class FootprintIntervalTests: XCTestCase {
    func testSamplerCoversAnAllocationFreedBeforeTheBoundary() {
        FootprintInterval.symbolMissingForTesting = true
        defer { FootprintInterval.stop(); FootprintInterval.symbolMissingForTesting = false }
        XCTAssertEqual(FootprintInterval.method, .sampled10ms)

        let bytes = 64 << 20
        FootprintInterval.reset()
        let before = Arena.processFootprint()
        // 64 MB, every page written, held 50 ms, unmapped: gone from the footprint at the boundary
        // (`mmap`, not `malloc`, whose freed large blocks can stay in the footprint).
        let block = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)!
        XCTAssertNotEqual(block, MAP_FAILED)
        memset(block, 1, bytes)
        usleep(50_000)
        munmap(block, bytes)
        usleep(20_000)
        let after = Arena.processFootprint()
        let peak = FootprintInterval.peak()
        XCTAssertLessThan(after, before + bytes / 2, "the block must be gone at the boundary for the test to mean anything")
        XCTAssertGreaterThanOrEqual(peak, before + bytes * 9 / 10, "the sampler missed the allocation")

        // A new interval forgets it.
        FootprintInterval.reset()
        usleep(20_000)
        XCTAssertLessThan(FootprintInterval.peak(), before + bytes / 2)

        // Stopped, nothing samples any more: the maximum is unmeasured.
        FootprintInterval.stop()
        XCTAssertEqual(FootprintInterval.maximum, 0)
    }
}
