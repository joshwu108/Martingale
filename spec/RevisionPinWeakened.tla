------------------------------ MODULE RevisionPinWeakened ------------------------------
(*
  WEAKENED variant of RevisionPin: the pin-before-draw ordering is removed.
  Actors can draw before pinning (or use the current_rev at draw time
  regardless of their pin). This yields a counterexample to PinInvariant.

  Counterexample trace:
    1. Learner publishes revision 0. current_rev = 0.
    2. Actor 0 enters PINNED state with pin=0.
    3. Learner publishes revision 1. current_rev = 1.
    4. Actor 0 draws action=0, but draw_rev is set to CURRENT_REV (=1) instead of pin (=0).
       This is the weakened step: draw_rev = current_rev, not actor_pin[a].
    5. Actor 0 appends: ledger record has revision=0 (from pin), pin_at_draw=1 (from draw).
       PinInvariant violated: revision (0) ≠ pin_at_draw (1).

  This file defines the WEAKENED spec with this counterexample.
*)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
  Actors, Revisions, Actions

ASSUME Actors = {0, 1}
ASSUME Revisions = {0, 1, 2}
ASSUME Actions = {0, 1}

VARIABLES
  actor_state, actor_pin, actor_draw, actor_draw_rev,
  ledger, published, current_rev

vars == <<actor_state, actor_pin, actor_draw, actor_draw_rev, ledger, published, current_rev>>

IDLE == "IDLE"
PINNED == "PINNED"
DREW == "DREW"
APPENDED == "APPENDED"

PinInvariant ==
  \A i \in 1..Len(ledger):
    ledger[i].revision = ledger[i].pin_at_draw

Init ==
  /\ actor_state = [a \in Actors |-> IDLE]
  /\ actor_pin = [a \in Actors |-> -1]
  /\ actor_draw = [a \in Actors |-> -1]
  /\ actor_draw_rev = [a \in Actors |-> -1]
  /\ ledger = <<>>
  /\ published = {0}
  /\ current_rev = 0

LearnerPublish ==
  /\ \E r \in Revisions \ published:
     /\ published' = published \cup {r}
     /\ current_rev' = r
     /\ UNCHANGED <<actor_state, actor_pin, actor_draw, actor_draw_rev, ledger>>

ActorPin(a) ==
  /\ actor_state[a] = IDLE
  /\ published # {}
  /\ actor_state' = [actor_state EXCEPT ![a] = PINNED]
  /\ actor_pin' = [actor_pin EXCEPT ![a] = current_rev]
  /\ UNCHANGED <<actor_draw, actor_draw_rev, ledger, published, current_rev>>

\* WEAKENED: draw_rev = current_rev (ignores pin — the bug)
ActorDrawWeakened(a, action) ==
  /\ actor_state[a] = PINNED
  /\ action \in Actions
  /\ actor_state' = [actor_state EXCEPT ![a] = DREW]
  /\ actor_draw' = [actor_draw EXCEPT ![a] = action]
  /\ actor_draw_rev' = [actor_draw_rev EXCEPT ![a] = current_rev]  \* BUG: uses current_rev, not pin
  /\ UNCHANGED <<actor_pin, ledger, published, current_rev>>

ActorAppend(a) ==
  /\ actor_state[a] = DREW
  /\ ledger' = Append(ledger, [
       actor |-> a,
       revision |-> actor_pin[a],
       pin_at_draw |-> actor_draw_rev[a],
       action |-> actor_draw[a]
     ])
  /\ actor_state' = [actor_state EXCEPT ![a] = APPENDED]
  /\ UNCHANGED <<actor_pin, actor_draw, actor_draw_rev, published, current_rev>>

ActorReset(a) ==
  /\ actor_state[a] = APPENDED
  /\ actor_state' = [actor_state EXCEPT ![a] = IDLE]
  /\ actor_pin' = [actor_pin EXCEPT ![a] = -1]
  /\ actor_draw' = [actor_draw EXCEPT ![a] = -1]
  /\ actor_draw_rev' = [actor_draw_rev EXCEPT ![a] = -1]
  /\ UNCHANGED <<ledger, published, current_rev>>

Next ==
  \/ LearnerPublish
  \/ \E a \in Actors: ActorPin(a)
  \/ \E a \in Actors, action \in Actions: ActorDrawWeakened(a, action)
  \/ \E a \in Actors: ActorAppend(a)
  \/ \E a \in Actors: ActorReset(a)

WeakenedSpec == Init /\ [][Next]_vars

\* PinInvariant is VIOLATED under WeakenedSpec (counterexample exists)
\* Counterexample: Learner publishes rev 0, Actor pins rev 0, Learner publishes rev 1,
\* Actor draws using current_rev=1 (not pin=0), Actor appends: revision=0 ≠ pin_at_draw=1.

================================================================================
