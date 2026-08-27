import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import LogLocator, NullFormatter

FDIR = "/home/n8/aps17/.claude/jobs/fd41db1b/tmp/fonts"
fontManager.addfont(f"{FDIR}/NotoSansKR-Regular.ttf")
fontManager.addfont(f"{FDIR}/NotoSansKR-Bold.ttf")
reg = FontProperties(fname=f"{FDIR}/NotoSansKR-Regular.ttf")
bold = FontProperties(fname=f"{FDIR}/NotoSansKR-Bold.ttf")

matplotlib.rcParams["svg.fonttype"] = "path"      # embed as outlines
matplotlib.rcParams["font.family"] = reg.get_name()

SURF = "#fcfcfb"
INK  = "#0b0b0b"
INK2 = "#52514e"
MUT  = "#8a8880"
SER  = "#2a78d6"
GRID = "#e6e5e1"

# (index, seq/s, adopted)  -- fast-node-normalised chain of measured A/B ratios.
D = [
    (0,   0.0333, True),  (1,   0.128,  True),  (2,   0.217,  True),
    (3,  21.0,    True),  (4,  42.5,    True),  (5, 161.4,    True),
    (6, 260.0,    True),  (7, 260.0,    False), (8, 270.7,    True),
    (9, 279.0,    True),  (10,300.4,    True),  (11,371.5,    True),
    (12,397.4,    True),  (13,437.2,    True),  (14,450.3,    True),
    (15,464.9,    True),  (16,471.7,    True),  (17,515.7,    True),
    (18,515.7,    False), (19,536.4,    True),  (21,542.1,    True),
    (22,542.1,    False), (23,550.0,    True),  (24,563.8,    True),
    (25,570.1,    True),  (26,578.9,    True),  (27,578.9,    False),
    (28,583.5,    True),  (29,583.5,    False), (30,583.5,    False),
    (31,583.5,    True),  (32,583.5,    True),  (33,586.5,    True),
    (34,605.6,    True),  (35,606.6,    True),  (36,609.4,    True),
    (37,615.1,    True),  (38,622.8,    True),  (39,622.8,    False),
    (40,638.4,    True),  (41,638.4,    False), (42,640.1,    True),
    (43,640.1,    False), (44,640.1,    False), (45,640.1,    False),
    (46,640.9,    True),  (47,640.9,    False), (48,640.9,    False),
    (49,640.9,    False), (50,640.9,    False), (51,640.9,    False),
    (52,640.9,    False), (53,640.9,    False), (54,640.9,    False),
    (55,643.5,    True),
]
xs = [d[0] for d in D]
ys = [d[1] for d in D]
ax_ = [d[0] for d in D if d[2]]
ay_ = [d[1] for d in D if d[2]]
rx_ = [d[0] for d in D if not d[2]]
ry_ = [d[1] for d in D if not d[2]]

fig = plt.figure(figsize=(7.6, 2.72), dpi=200)
fig.patch.set_facecolor(SURF)
ax = fig.add_axes([0.064, 0.155, 0.924, 0.805])
ax.set_facecolor(SURF)

ax.step(xs, ys, where="post", color=SER, lw=1.6, solid_capstyle="round", zorder=3)
ax.plot(rx_, ry_, "o", ms=4.2, mfc=SURF, mec=MUT, mew=1.1, zorder=4,
        label="기각 · 코드 원복")
ax.plot(ax_, ay_, "o", ms=4.2, mfc=SER, mec=SURF, mew=0.9, zorder=5,
        label="채택")

ax.set_yscale("log")
ax.set_ylim(0.011, 2600)
ax.set_xlim(-1.6, 61.0)
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000])
ax.set_yticklabels(["0.01", "0.1", "1", "10", "100", "1000"])
ax.set_xticks(list(range(0, 56, 5)))

ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.grid(axis="y", which="minor", color=GRID, lw=0.3, alpha=0.55, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
    ax.spines[s].set_linewidth(0.7)
ax.tick_params(colors=INK2, labelsize=6.6, length=2.4, width=0.7)
ax.set_xlabel("실험 인덱스 (EXP)", fontproperties=reg, fontsize=7.0, color=INK2, labelpad=1.5)
ax.set_ylabel("seq/s (log)", fontproperties=reg, fontsize=7.0, color=INK2, labelpad=2)

ANN = [
    (0,  0.0333, "000 baseline 0.033",        (2,   7), "left",   "bottom"),
    (1,  0.128,  "001 배치화 2.88×",           (3,  -8), "left",   "top"),
    (3,  21.0,   "003 MoE→GPU 35.9×",         (6,  -9), "left",   "top"),
    (5,  161.4,  "005 레지스터 타일 3.77×",     (5,  -9), "left",   "top"),
    (6,  260.0,  "006 attention 1.61×",       (3,  11), "left",   "bottom"),
    (11, 371.5,  "011 prefix trie 1.24×",     (4,  -8), "left",   "top"),
    (17, 515.7,  "017 bank conflict 1.09×",   (-2, 11), "center", "bottom"),
    (34, 605.6,  "034 GQA 블록",              (0,  12), "center", "bottom"),
    (40, 638.4,  "040 마지막 층 1024행",       (2,  12), "left",   "bottom"),
    (55, 643.5,  "055\n643.5",                 (5,   0), "left",   "center"),
]
for x, y, t, (dx, dy), ha, va in ANN:
    ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontproperties=reg, fontsize=5.7,
                color=INK, linespacing=1.25, zorder=6)

leg = ax.legend(loc="lower left", bbox_to_anchor=(0.145, 0.075), frameon=False,
                handletextpad=0.35, borderpad=0.1, labelspacing=0.28,
                prop={"family": reg.get_name(), "size": 6.2})
for t in leg.get_texts():
    t.set_color(INK2)

# inset: linear view of the long tail, where log scale flattens everything
ins = fig.add_axes([0.560, 0.250, 0.295, 0.320])
ins.set_facecolor("#f4f3f0")
tail = [d for d in D if d[0] >= 16]
ins.step([d[0] for d in tail], [d[1] for d in tail], where="post",
         color=SER, lw=1.2, zorder=3)
ins.plot([d[0] for d in tail if not d[2]], [d[1] for d in tail if not d[2]],
         "o", ms=2.6, mfc="#f4f3f0", mec=MUT, mew=0.8, zorder=4)
ins.plot([d[0] for d in tail if d[2]], [d[1] for d in tail if d[2]],
         "o", ms=2.6, mfc=SER, mec="#f4f3f0", mew=0.6, zorder=5)
ins.set_xlim(15.2, 56)
ins.set_ylim(455, 675)
ins.set_yticks([475, 550, 625])
ins.set_xticks([20, 30, 40, 50])
ins.tick_params(colors=INK2, labelsize=5.2, length=1.6, width=0.5, pad=1.5)
ins.grid(axis="y", color="#dedcd7", lw=0.45, zorder=0)
for s in ("top", "right"):
    ins.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ins.spines[s].set_color("#dedcd7")
    ins.spines[s].set_linewidth(0.5)
ins.set_title("EXP-016~055 선형 확대", fontproperties=reg, fontsize=5.6,
              color=INK2, pad=2.0)

fig.savefig("/home/n8/aps17/.claude/jobs/fd41db1b/tmp/chart.svg",
            facecolor=SURF, format="svg")
fig.savefig("/home/n8/aps17/.claude/jobs/fd41db1b/tmp/chart.png",
            facecolor=SURF, dpi=200)
print("saved")
print("total speedup: %.0fx" % (643.5 / 0.0333))
