"""Methods schematic of the cascade-attribution ledger (SI S5, eq. cascade).

Writes cascade_attribution_figure.svg next to this file. Convert with
    rsvg-convert -f pdf -o cascade_attribution_figure.pdf cascade_attribution_figure.svg

One panel, one column (88 mm). Lanes are agents, time runs right. Teal bars are
vegetarian stints (their length T is the credit unit, truncated at t_end), dots are
conversion events, the hollow dot a reversion. Purple arrows carry credit from a
conversion up to the sources in the converter's memory buffer M, split by exposure
share sigma; a source forwards along its own bar to its own conversion event, times
lambda per hop. The walk shown is the one seeded by k's conversion; j's own credit
T_j enters the same channels.
"""

VEG, OMN, CREDIT, INK, GREY = "#2a9d8f", "#c9c9c9", "#7E3F98", "#222222", "#888888"
FONT = "font-family=\"Helvetica, Arial, sans-serif\""

X0, XEND, XFAR = 20, 222, 240            # time origin, t_end, right edge of faded bars
LANES = {"z": 22, "a": 46, "j": 70, "k": 94, "b": 118}
H = 3.5                                   # half-height of a stint bar

# (agent, start, end, parent, sigma); end=None runs past t_end
STINTS = [("z", X0, None), ("b", X0, None), ("a", 60, None), ("j", 110, 175), ("k", 145, None)]
LINKS = [("a", "z", 60, 1.0), ("j", "a", 110, 0.55), ("j", "b", 110, 0.45), ("k", "j", 145, 1.0)]

out = []
w = out.append


def text(x, y, s, size=6, fill=INK, anchor="middle", style=""):
    w(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" {FONT}{style}>{s}</text>')


def sub(base, s):
    return f'{base}<tspan font-size="4.5" dy="1.5">{s}</tspan>'


w('<svg xmlns="http://www.w3.org/2000/svg" width="88mm" height="60.7mm" viewBox="0 0 250 172.5">')
w(f'<defs><marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
  f'<polygon points="0 0.5,6 3,0 5.5" fill="{CREDIT}"/></marker></defs>')

# lanes and agent labels
for name, y in LANES.items():
    w(f'<line x1="{X0}" y1="{y}" x2="{XFAR}" y2="{y}" stroke="#e6e6e6" stroke-width="0.6"/>')
    text(11, y + 2.5, name, size=8, style=' font-style="italic"')

# stints: solid to t_end, faded beyond (truncation)
for name, s, e in STINTS:
    y = LANES[name] - H
    if e is None:
        w(f'<rect x="{s}" y="{y}" width="{XEND - s}" height="{2 * H}" fill="{VEG}"/>')
        w(f'<rect x="{XEND}" y="{y}" width="{XFAR - XEND}" height="{2 * H}" fill="{VEG}" opacity="0.3"/>')
    else:
        w(f'<rect x="{s}" y="{y}" width="{e - s}" height="{2 * H}" fill="{VEG}"/>')

# t_end
w(f'<line x1="{XEND}" y1="12" x2="{XEND}" y2="132" stroke="{INK}" stroke-width="0.6" stroke-dasharray="2.5,1.8"/>')
text(XEND, 9, 't<tspan font-size="4.5" dy="1.5" font-style="normal">end</tspan>', size=6.5, style=' font-style="italic"')

# credit forwarding along a bar to the agent's own conversion event, x lambda per hop
for name, x_in, x_ev in [("j", 145, 110), ("a", 110, 60)]:
    y = LANES[name]
    w(f'<line x1="{x_in}" y1="{y}" x2="{x_ev + 2}" y2="{y}" stroke="{CREDIT}" stroke-width="1.1" stroke-dasharray="1.2,1.6"/>')
    text((x_in + x_ev) / 2, y - 5.5, "&#215;&#955;", size=6, fill=CREDIT, style=' font-style="italic"')

# credit arrows: child event -> parent bar, width by share
for child, parent, x, sigma in LINKS:
    y1, y2 = LANES[child] - H - 0.5, LANES[parent] + H + 0.8
    if LANES[parent] > LANES[child]:            # parent lane below (b)
        y1, y2 = LANES[child] + H + 0.5, LANES[parent] - H - 0.8
    w(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{CREDIT}" stroke-width="{0.9 + 1.2 * sigma:.2f}" marker-end="url(#arr)"/>')

# share labels at j's split
text(114, 60, sub("&#963;", "a"), size=6, fill=CREDIT, anchor="start", style=' font-style="italic"')
text(114, 106, sub("&#963;", "b"), size=6, fill=CREDIT, anchor="start", style=' font-style="italic"')
text(149, 84, sub("T", "k"), size=6, fill=CREDIT, anchor="start", style=' font-style="italic"')

# memory buffer of j at conversion: 9 samples, sources labelled
cells = ["", "a", "", "", "b", "a", "", "", ""]
cx, cy, cs = 58, 57.5, 5
text(cx - 2.5, cy + 4.3, "M", size=6, anchor="end", style=' font-style="italic"')
for i, src in enumerate(cells):
    x = cx + i * cs
    w(f'<rect x="{x}" y="{cy}" width="{cs - 0.7}" height="{cs - 0.7}" fill="{VEG if src else OMN}"/>')
    if src:
        text(x + (cs - 0.7) / 2, cy + 3.6, src, size=4.2, fill="#fff", style=' font-style="italic"')
w(f'<path d="M {cx + 9 * cs - 1.5},{cy + cs / 2} L 106,{cy + cs / 2} L 108.5,{LANES["j"] - H - 1.5}" fill="none" stroke="{GREY}" stroke-width="0.5"/>')

# events
for name, s, e in STINTS:
    if s > X0:
        w(f'<circle cx="{s}" cy="{LANES[name]}" r="2.6" fill="{INK}"/>')
    if e is not None:
        w(f'<circle cx="{e}" cy="{LANES[name]}" r="2.6" fill="#fff" stroke="{INK}" stroke-width="1"/>')

# stint lengths T, written on the bars (bar length is the credit unit)
for name, x in [("j", 163), ("k", 207)]:
    text(x, LANES[name] + 2, sub("T", name), size=5.5, fill="#fff", style=' font-style="italic"')

# time axis
w(f'<line x1="{X0}" y1="136" x2="{XFAR}" y2="136" stroke="{INK}" stroke-width="0.6"/>')
w(f'<polygon points="{XFAR},133.5 {XFAR + 4},136 {XFAR},138.5" fill="{INK}"/>')
text(XFAR + 6, 138.3, "t", size=7, anchor="start", style=' font-style="italic"')

# legend
ly = 158
w(f'<rect x="{X0}" y="{ly - 3}" width="12" height="6" fill="{VEG}"/>')
text(X0 + 15, ly + 2, "vegetarian stint", size=5.5, anchor="start")
w(f'<circle cx="{X0 + 60}" cy="{ly}" r="2.4" fill="{INK}"/>')
text(X0 + 65, ly + 2, "conversion", size=5.5, anchor="start")
w(f'<circle cx="{X0 + 98}" cy="{ly}" r="2.4" fill="#fff" stroke="{INK}" stroke-width="1"/>')
text(X0 + 103, ly + 2, "reversion", size=5.5, anchor="start")
w(f'<line x1="{X0 + 132}" y1="{ly}" x2="{X0 + 144}" y2="{ly}" stroke="{CREDIT}" stroke-width="1.6" marker-end="url(#arr)"/>')
text(X0 + 148, ly + 2, "credit", size=5.5, anchor="start")
w(f'<rect x="{X0 + 168}" y="{ly - 2.5}" width="5" height="5" fill="{OMN}"/>')
text(X0 + 176, ly + 2, "omnivore sample", size=5.5, anchor="start")

w("</svg>")

import pathlib
pathlib.Path(__file__).with_suffix(".svg").write_text("\n".join(out) + "\n")
