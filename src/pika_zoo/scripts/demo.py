"""
Demo script — watch or play Pikachu Volleyball matches.

Usage:
    uv run pika-zoo-demo                              # AI vs AI
    uv run pika-zoo-demo --p1 human                   # Human vs AI
    uv run pika-zoo-demo --p1 human --p2 human        # Human vs Human
    uv run pika-zoo-demo --winning-score 5 --seed 123

Controls:
    Player 1: D(left) G(right) R(up) V(down) Z(power hit)
    Player 2: Arrow keys + Enter (power hit)
"""

from __future__ import annotations

import argparse

import pygame

from pika_zoo.ai.registry import get_ai
from pika_zoo.env import env
from pika_zoo.scripts.keyboard import get_action_from_keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch or play Pikachu Volleyball")
    parser.add_argument("--winning-score", type=int, default=15, help="Score to win (default: 15)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--p1", type=str, default="builtin", help="Player 1: AI name or 'human' (default: builtin)")
    parser.add_argument("--p2", type=str, default="builtin", help="Player 2: AI name or 'human' (default: builtin)")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    args = parser.parse_args(argv)

    p1_human = args.p1 == "human"
    p2_human = args.p2 == "human"

    ai_policies = {}
    if not p1_human:
        ai_policies["player_1"] = get_ai(args.p1)
    if not p2_human:
        ai_policies["player_2"] = get_ai(args.p2)

    e = env(
        render_mode="human",
        ai_policies=ai_policies,
        winning_score=args.winning_score,
    )
    e.reset(seed=args.seed)

    mode = []
    if p1_human:
        mode.append("Human")
    else:
        mode.append(args.p1)
    mode.append("vs")
    if p2_human:
        mode.append("Human")
    else:
        mode.append(args.p2)
    print(f"Pikachu Volleyball — {' '.join(mode)} (first to {args.winning_score})")
    if p1_human:
        print("  P1: D(left) G(right) R(up) V(down) Z(power hit)")
    if p2_human:
        print("  P2: Arrow keys + Enter (power hit)")

    # Trigger lazy pygame init via first render
    e.render()

    frame = 0
    running = True
    while running:
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

        if not running:
            break

        # Get actions
        actions = {"player_1": 0, "player_2": 0}
        if p1_human or p2_human:
            keys = pygame.key.get_pressed()
            if p1_human:
                actions["player_1"] = get_action_from_keys(keys, "player_1")
            if p2_human:
                actions["player_2"] = get_action_from_keys(keys, "player_2")

        obs, rewards, terms, truncs, infos = e.step(actions)
        e.render()

        frame += 1

        if any(terms.values()):
            scores = infos["player_1"]["scores"]
            winner = "Player 1" if scores[0] > scores[1] else "Player 2"
            print(f"Game over! {winner} wins {scores[0]}-{scores[1]} ({frame} frames)")
            break

    e.close()


if __name__ == "__main__":
    main()
