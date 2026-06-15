from collections import deque
from dataclasses import dataclass
import argparse


# moves = {
#     0: [1, 0, 0, 0, 0, 1],
#     1: [1, 1, -1, 0, 0, 0],
#     2: [1, 0, 1, -1, -1, 0],
#     3: [0, -1, 0, 1, 0, 1],
#     4: [0, 0, 0, 0, 1, 0],
#     5: [0, 0, 0, 1, -1, 1],
# }


moves = {
    5: [1, 0, 0, 0, 0, 1],
    4: [0, 0, 0, -1, 1, 1],
    3: [0, -1, -1, 1, 0, 1],
    2: [1, 0, 1, 0, -1, 0],
    1: [0, 1, 0, 0, 0, 0],
    0: [1, -1, 1, 0, 0, 0],
}


# initial = (4, 4, 6, 4, 3, 1)
initial = (1, 3, 4, 6, 4, 4)
target = (3, 3, 3, 3, 3, 3)


@dataclass(frozen=True, init=True)
class State:
    curr: tuple[int, ...]
    hist: tuple[tuple[int, int], ...]


max_moves = 45
n_holes = 7


def solve(
    start: tuple[int, ...],
    goal: tuple[int, ...],
    moves_map: dict[int, list[int]],
    limit_moves: int = max_moves,
    n_pins: int = n_holes,
) -> tuple[tuple[int, int], ...]:
    q: deque[State] = deque()
    q.append(State(curr=start, hist=()))
    visited: set[tuple[int, ...]] = set()

    while q:
        tmp = q.popleft()
        if tmp.curr == goal:
            print(tmp.hist)
            return tmp.hist

        if tmp.curr in visited:
            continue

        if len(tmp.hist) > limit_moves:
            continue

        for move_idx, pattern in moves_map.items():
            for direction in (1, -1):
                for shift in range(len(pattern)):
                    nxt = tuple(
                        map(sum, zip(tmp.curr, (shift * direction * o for o in pattern)))
                    )
                    if any(x >= n_pins or x < 0 for x in nxt):
                        break
                    q.append(State(curr=nxt, hist=tmp.hist + ((move_idx, direction * shift),)))
        visited.add(tmp.curr)

    return ()


def apply_move(curr: tuple[int, ...], move: tuple[int, int]) -> tuple[int, ...]:
    move_idx, signed_shift = move
    pattern = moves[move_idx]
    return tuple(map(sum, zip(curr, (signed_shift * o for o in pattern))))


def build_trajectory(
    start: tuple[int, ...],
    history: tuple[tuple[int, int], ...],
) -> list[tuple[int, ...]]:
    states = [start]
    curr = start
    for mv in history:
        curr = apply_move(curr, mv)
        states.append(curr)
    return states


def format_dir(move: tuple[int, int]) -> str:
    move_idx, signed_shift = move
    if signed_shift < 0:
        return f"[{move_idx}] R->{-signed_shift}"
    return f"[{move_idx}] L->{signed_shift}"


def render_state(
    state: tuple[int, ...],
    step: int,
    total: int,
    move: tuple[int, int] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("Locker/Pins visualization")
    lines.append(f"Step {step}/{total}")
    lines.append("Pins:    " + " ".join(str(i) for i in range(n_holes)))

    for locker_idx, pin_idx in enumerate(state):
        cells = ["."] * n_holes
        cells[pin_idx] = "O"
        lines.append(f"L{locker_idx}:     " + " ".join(cells) + f"   (pin={pin_idx})")

    lines.append("Move:    " + ("START" if move is None else format_dir(move)))
    lines.append("")
    lines.append("Controls: n=next, p=prev, q=quit")
    return "\n".join(lines)


def interactive_replay(
    states: list[tuple[int, ...]],
    history: tuple[tuple[int, int], ...],
) -> None:
    idx = 0
    total = len(states) - 1
    while True:
        # ANSI clear screen + cursor to top-left.
        print("\033[2J\033[H", end="")
        move = history[idx - 1] if idx > 0 else None
        print(render_state(states[idx], idx, total, move))
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "n":
            idx = min(total, idx + 1)
        elif cmd == "p":
            idx = max(0, idx - 1)
        elif cmd.isdigit():
            idx = max(0, min(total, int(cmd)))


def interactive_plot(
    states: list[tuple[int, ...]],
    history: tuple[tuple[int, int], ...],
    mode: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --graphical mode. Install it with: pip install matplotlib"
        ) from exc

    n_lockers = len(states[0])
    n_steps = len(states) - 1
    locker_indices = list(range(n_lockers))
    step_indices = list(range(n_steps + 1))
    locker_paths = [[st[locker] for st in states] for locker in locker_indices]

    fig = plt.figure(figsize=(12, 8))
    current_step = {"idx": 0}

    if mode == "3d":
        ax = fig.add_subplot(111, projection="3d")
        for locker in locker_indices:
            ax.plot(
                step_indices,
                [locker] * len(step_indices),
                locker_paths[locker],
                linewidth=2,
                label=f"L{locker}",
            )

        current_points = ax.scatter(
            [0] * n_lockers,
            locker_indices,
            [states[0][locker] for locker in locker_indices],
            s=100,
            c="red",
            depthshade=True,
        )

        ax.set_xlabel("Step")
        ax.set_ylabel("Locker")
        ax.set_zlabel("Pin")
        ax.set_xlim(-0.5, n_steps + 0.5)
        ax.set_ylim(-0.5, n_lockers - 0.5)
        ax.set_zlim(-0.5, n_holes - 0.5)
        title = ax.set_title("3D Transition View | Step 0")
        ax.legend(loc="upper left")
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.55)
        fig.subplots_adjust(top=0.93, bottom=0.18)
        ax_board = fig.add_subplot(gs[0])
        ax_paths = fig.add_subplot(gs[1])

        ax_board.set_xlim(-0.5, n_holes - 0.5)
        ax_board.set_ylim(-0.5, n_lockers - 0.5)
        ax_board.set_xticks(range(n_holes))
        ax_board.set_yticks(locker_indices)
        ax_board.set_xlabel("Pin")
        ax_board.set_ylabel("Locker")
        ax_board.set_title("Current state")
        ax_board.grid(True, alpha=0.3)

        board_points = ax_board.scatter(
            [states[0][locker] for locker in locker_indices],
            locker_indices,
            s=260,
            c=locker_indices,
            cmap="tab10",
            edgecolors="black",
        )

        ax_paths.set_xlim(0, n_steps)
        ax_paths.set_ylim(-0.5, n_holes - 0.5)
        ax_paths.set_xlabel("Step")
        ax_paths.set_ylabel("Pin")
        ax_paths.set_title("Per-locker pin trajectory")
        ax_paths.grid(True, alpha=0.3)

        for locker in locker_indices:
            ax_paths.plot(step_indices, locker_paths[locker], linewidth=2, label=f"L{locker}")
        current_line = ax_paths.axvline(0, color="red", linestyle="--", linewidth=2)

        move_text = ax_board.text(
            0.02,
            1.02,
            "Step 0/0 | START",
            transform=ax_board.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        ax_paths.legend(ncols=3, loc="upper right")

    slider_ax = fig.add_axes([0.15, 0.04, 0.52, 0.03])
    prev_ax = fig.add_axes([0.70, 0.035, 0.08, 0.05])
    next_ax = fig.add_axes([0.80, 0.035, 0.08, 0.05])
    reset_ax = fig.add_axes([0.90, 0.035, 0.08, 0.05])

    slider = Slider(slider_ax, "Step", 0, n_steps, valinit=0, valstep=1)
    btn_prev = Button(prev_ax, "Prev")
    btn_next = Button(next_ax, "Next")
    btn_reset = Button(reset_ax, "Reset")

    def step_label(idx: int) -> str:
        if idx == 0:
            return "START"
        return format_dir(history[idx - 1])

    def update(step: int) -> None:
        idx = int(step)
        current_step["idx"] = idx

        if mode == "3d":
            points = [states[idx][locker] for locker in locker_indices]
            current_points._offsets3d = ([idx] * n_lockers, locker_indices, points)
            title.set_text(f"3D Transition View | Step {idx}/{n_steps} | {step_label(idx)}")
        else:
            board_points.set_offsets([[states[idx][locker], locker] for locker in locker_indices])
            current_line.set_xdata([idx, idx])
            move_text.set_text(f"Step {idx}/{n_steps} | {step_label(idx)}")

        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_prev(_event) -> None:
        slider.set_val(max(0, current_step["idx"] - 1))

    def on_next(_event) -> None:
        slider.set_val(min(n_steps, current_step["idx"] + 1))

    def on_reset(_event) -> None:
        slider.set_val(0)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_reset.on_clicked(on_reset)

    def on_key(event) -> None:
        if event.key == "left":
            on_prev(None)
        elif event.key == "right":
            on_next(None)
        elif event.key == "home":
            on_reset(None)

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    plt.show()


def save_gif(
    states: list[tuple[int, ...]],
    history: tuple[tuple[int, int], ...],
    path: str,
    fps: int = 2,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    n_lockers = len(states[0])
    n_steps = len(states) - 1
    locker_indices = list(range(n_lockers))
    step_indices = list(range(n_steps + 1))
    locker_paths = [[st[locker] for st in states] for locker in locker_indices]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.55)
    fig.subplots_adjust(top=0.93, bottom=0.10)
    ax_board = fig.add_subplot(gs[0])
    ax_paths = fig.add_subplot(gs[1])

    ax_board.set_xlim(-0.5, n_holes - 0.5)
    ax_board.set_ylim(-0.5, n_lockers - 0.5)
    ax_board.set_xticks(range(n_holes))
    ax_board.set_yticks(locker_indices)
    ax_board.set_xlabel("Pin")
    ax_board.set_ylabel("Locker")
    ax_board.set_title("Current state")
    ax_board.grid(True, alpha=0.3)

    board_points = ax_board.scatter(
        [states[0][locker] for locker in locker_indices],
        locker_indices,
        s=260,
        c=locker_indices,
        cmap="tab10",
        edgecolors="black",
    )

    ax_paths.set_xlim(0, n_steps)
    ax_paths.set_ylim(-0.5, n_holes - 0.5)
    ax_paths.set_xlabel("Step")
    ax_paths.set_ylabel("Pin")
    ax_paths.set_title("Per-locker pin trajectory")
    ax_paths.grid(True, alpha=0.3)

    for locker in locker_indices:
        ax_paths.plot(step_indices, locker_paths[locker], linewidth=2, label=f"L{locker}")
    current_line = ax_paths.axvline(0, color="red", linestyle="--", linewidth=2)
    ax_paths.legend(ncols=3, loc="upper right")

    move_text = ax_board.text(
        0.02, 1.02, "Step 0 | START",
        transform=ax_board.transAxes, fontsize=11, fontweight="bold",
    )

    def animate(frame: int):
        label = "START" if frame == 0 else format_dir(history[frame - 1])
        board_points.set_offsets([[states[frame][locker], locker] for locker in locker_indices])
        current_line.set_xdata([frame, frame])
        move_text.set_text(f"Step {frame}/{n_steps} | {label}")
        return board_points, current_line, move_text

    anim = FuncAnimation(fig, animate, frames=n_steps + 1, interval=1000 // fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve and visualize locker pin moves.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Replay the found solution step-by-step in the terminal.",
    )
    parser.add_argument(
        "--graphical",
        choices=("2d", "3d"),
        help="Open a graphical interactive replay window.",
    )
    parser.add_argument(
        "--save-gif",
        metavar="FILE",
        help="Save 2D step animation as a GIF (e.g. solution.gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the saved GIF (default: 2).",
    )
    args = parser.parse_args()

    history = solve(initial, target, moves)
    if not history:
        print("No solution found.")
        return

    if args.interactive:
        interactive_replay(build_trajectory(initial, history), history)
        return

    if args.graphical:
        interactive_plot(build_trajectory(initial, history), history, args.graphical)
        return

    if args.save_gif:
        save_gif(build_trajectory(initial, history), history, args.save_gif, fps=args.fps)
        return

    print([format_dir(move) for move in history])


if __name__ == "__main__":
    main()