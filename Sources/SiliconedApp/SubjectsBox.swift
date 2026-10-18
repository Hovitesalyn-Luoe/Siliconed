// **The named subjects in the rack** (experimental, `Subjects.swift`): the Images box in its Subjects
// mode, and the row under the prompt that says what the subjects make of it.
//
//     ┌ Images ───────────────── 2/3 ┐   2 cited of the 3 a render takes
//     │ [ Images | Subjects ]        │   one or the other, never both
//     │ (▣ @alice) (▣ @red-coat) (+) │   one chip per subject — a click cites it, its menu edits or
//     └──────────────────────────────┘   deletes it; an image dropped here, or « + », names a new one
//
//     Prompt … @alice in @red-coat
//     1 ● 2 ●  2 images     Model Prompt     ← the model's prompt, in a popover; or why it is refused

import AppKit
import Siliconed
import SwiftUI

/// **The library of subjects, the Images box in its Subjects mode** — in memory, as many as one wants;
/// the prompt cites 3 at most. A subject cited is one image to the model, numbered by first appearance.
struct SubjectShelf: View {
    let app: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.vertical) {
                Flow(spacing: 5) {
                    ForEach(app.subjects.all) { s in
                        Button { app.cite(s) } label: {
                            SubjectChip(name: s.name, kind: s.kind, thumbnail: app.subjects.thumbnail(s), pictures: s.pictures.count)
                        }
                        .buttonStyle(.plain)
                        .help(Text(verbatim: "\(s.kind.label) — \(s.kind.taken)"))
                        .draggable("@\(s.name) ")
                        .contextMenu {
                            Button("Cite in the Prompt") { app.cite(s) }
                            Button("Edit…") { app.editSubject(s) }
                            Divider()
                            Button("Delete…", role: .destructive) { app.confirmDeletion(s) }
                        }
                    }
                    Button { app.chooseSubjectImage() } label: {
                        Capsule()
                            .strokeBorder(style: StrokeStyle(lineWidth: 1.25, dash: [4, 3]))
                            .foregroundStyle(WireKind.image.color.opacity(0.7))
                            .overlay(Image(systemName: "plus").font(.caption).foregroundStyle(WireKind.image.color))
                            .frame(width: 36, height: 24)
                            .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Name a subject from an image — or drop it here")
                    .accessibilityLabel(Text("Name a Subject"))
                }
            }
            .frame(maxHeight: 132)
            .fixedSize(horizontal: false, vertical: true)
            Text(app.subjects.all.isEmpty
                 ? "Drop an image to name a person, a garment, an object, a place or a style, then write @name in the prompt."
                 : "Click a subject to cite it. A render takes 3 images at most.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// **Under the prompt, what the subjects make of it**: the images in their order — or why the prompt
/// is refused —, and the model's prompt, hidden until asked for (it serves to understand a miss).
struct ModelPromptRow: View {
    let app: AppState
    let rewrite: Result<NamedSubjects.Rewritten, NamedSubjects.Failure>
    @State private var shown = false

    var body: some View {
        HStack(spacing: 6) {
            switch rewrite {
            case .failure(let f):
                Image(systemName: "exclamationmark.circle").foregroundStyle(Palette.failure)
                Text(Problem(f).title).foregroundStyle(.secondary)
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            case .success(let r):
                ForEach(r.placed, id: \.number) { p in
                    HStack(spacing: 3) {
                        Text(verbatim: "\(p.number)").monospacedDigit().foregroundStyle(.secondary)
                        Circle().fill(p.subject.kind.color).frame(width: 6, height: 6)
                    }
                    .help(Text(verbatim: "@\(p.subject.name) — \(p.subject.kind.label)"))
                }
                Text(r.placed.count == 1 ? String(localized: "1 image") : String(localized: "\(r.placed.count) images"))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 4)
                Button("Model Prompt") { shown.toggle() }
                    .buttonStyle(.link)
                    .help("What the model reads: the prompt rewritten, each subject tied to its image")
                    .popover(isPresented: $shown, arrowEdge: .trailing) { ModelPromptPopover(app: app, rewrite: r) }
            }
        }
        .font(.caption)
        .padding(.top, 6)
    }
}

private struct ModelPromptPopover: View {
    let app: AppState
    let rewrite: NamedSubjects.Rewritten

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("What the Model Reads").font(.headline)
            ScrollView {
                Text(rewrite.prompt)
                    .font(.callout).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxHeight: 220)
            Divider()
            HStack(alignment: .top, spacing: 10) {
                ForEach(rewrite.placed, id: \.number) { p in
                    VStack(spacing: 4) {
                        Group {
                            if let t = app.subjects.thumbnail(p.subject, picture: p.subject.pictures.firstIndex(of: p.picture) ?? 0) {
                                Image(nsImage: t).resizable().aspectRatio(contentMode: .fill)
                            } else {
                                Color.secondary.opacity(0.2)
                            }
                        }
                        .frame(width: 64, height: 64)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(p.subject.kind.color, lineWidth: 1.5))
                        Text("image \(p.number)").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            Text("The wording is provisional: it will be settled by eye on a series of renders.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(width: 380)
    }
}

/// **Chips on as many lines as they need**, left to right.
private struct Flow: Layout {
    var spacing: CGFloat = 6

    private typealias Row = (items: [Int], width: CGFloat, height: CGFloat)

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? .infinity, subviews)
        let width = rows.map(\.width).max() ?? 0
        let height = rows.reduce(0) { $0 + $1.height } + spacing * CGFloat(max(rows.count - 1, 0))
        return CGSize(width: proposal.width ?? width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(bounds.width, subviews) {
            var x = bounds.minX
            for i in row.items {
                let size = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: y + (row.height - size.height) / 2), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> [Row] {
        var rows: [Row] = []
        var current: Row = ([], 0, 0)
        for (i, v) in subviews.enumerated() {
            let s = v.sizeThatFits(.unspecified)
            let w = min(s.width, width)
            if !current.items.isEmpty && current.width + spacing + w > width {
                rows.append(current)
                current = ([], 0, 0)
            }
            current.width += (current.items.isEmpty ? 0 : spacing) + w
            current.height = max(current.height, s.height)
            current.items.append(i)
        }
        if !current.items.isEmpty { rows.append(current) }
        return rows
    }
}
