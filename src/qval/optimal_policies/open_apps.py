"""Scripted policies for OpenApps tasks.

Each policy maps the flattened accessibility tree of the current page to
the next BrowserGym action that advances the task toward completion.

If the policy cannot determine a forward action (required element not
visible, navigation-recovery failed, dispatcher couldn't identify the
task, etc.) it raises :class:`PolicyStuckError` instead of emitting a
filler ``noop`` — silently looping noops would burn ``max_steps`` and
hide real failures, whereas an exception aborts the rollout immediately
so the bug is visible.

BrowserGym assigns element bids per-session, so we locate elements by
role + visible name rather than hard-coded bids.

Registered policy names follow the pattern ``open_apps_<task_name>``,
mirroring the ``frozen_lake_4x4`` / ``frozen_lake_8x8`` convention.
Each policy's metadata stores ``task_name`` so the dispatch function
can branch on it.
"""

from __future__ import annotations

import re
from typing import Any

from qval.optimal_policies import register_policy


# Regex that captures the canonical axtree line shape:
#   "[<bid>] <role> '<name>'<trailing attrs>"
# where the trailing attrs (after the name's closing quote) are things
# like ", clickable, visible, focused".  The role is always a single
# token with no spaces.
_AXTREE_LINE = re.compile(
    r"^\s*\[(?P<bid>\d+)\]\s+(?P<role>\S+)\s+'(?P<name>[^']*)'(?P<attrs>[^\n]*)$"
)


# ---------------------------------------------------------------------------
# Axtree parsing helpers
# ---------------------------------------------------------------------------


def _iter_entries(axtree: str):
    """Yield (line_index, bid, role, name, attrs) for each element line."""
    for idx, line in enumerate(axtree.splitlines()):
        m = _AXTREE_LINE.match(line)
        if m:
            yield idx, m.group("bid"), m.group("role"), m.group("name"), m.group("attrs")


def _find_bid(
    axtree: str,
    role: str,
    name_contains: str,
    *,
    require_visible: bool = True,
    require_clickable: bool = False,
) -> str | None:
    """Return the first bid whose role matches and whose name contains the text.

    Matching is case-insensitive on the name substring.  Pass
    ``require_clickable=True`` to skip purely-visible entries.
    """
    needle = name_contains.lower()
    for _, bid, r, name, attrs in _iter_entries(axtree):
        if r != role:
            continue
        if needle and needle not in name.lower():
            continue
        if require_visible and "visible" not in attrs:
            continue
        if require_clickable and "clickable" not in attrs:
            continue
        return bid
    return None


def _find_bid_near_static_text(
    axtree: str,
    static_text: str,
    role: str,
    *,
    look_back: int = 6,
    extra_filter: str | None = None,
) -> str | None:
    """Locate an element of ``role`` sitting just above a ``StaticText 'X'`` line.

    OpenApps list items render as::

        [N] listitem '', visible
            [N+1] checkbox '', clickable, visible, checked='false'
            StaticText 'Todo name'
            [N+2] button 'Edit', clickable, visible

    so the checkbox is the most recent ``role == "checkbox"`` line
    appearing above the StaticText.  ``look_back`` caps how far above the
    anchor we scan.  ``extra_filter`` is a substring the matched line
    must contain (e.g. ``"checked='false'"``).
    """
    lines = axtree.splitlines()
    anchor_marker = f"StaticText '{static_text}'"
    for idx, line in enumerate(lines):
        if anchor_marker in line:
            for j in range(idx - 1, max(-1, idx - 1 - look_back), -1):
                m = _AXTREE_LINE.match(lines[j])
                if m and m.group("role") == role:
                    if extra_filter and extra_filter not in lines[j]:
                        continue
                    return m.group("bid")
    return None


def _find_exact_bid(
    axtree: str,
    role: str,
    name_exact: str,
    *,
    require_visible: bool = True,
    require_clickable: bool = False,
) -> str | None:
    """Return the first bid whose role and visible name exactly match."""
    for _, bid, r, name, attrs in _iter_entries(axtree):
        if r != role or name != name_exact:
            continue
        if require_visible and "visible" not in attrs:
            continue
        if require_clickable and "clickable" not in attrs:
            continue
        return bid
    return None


def _has_entry(
    axtree: str,
    *,
    role: str | None = None,
    name_contains: str | None = None,
    name_exact: str | None = None,
    attrs_contains: str | None = None,
    require_visible: bool = True,
    require_clickable: bool = False,
) -> bool:
    """Predicate version of the AXTree element scan used by classifiers."""
    needle = name_contains.lower() if name_contains is not None else None
    for _, _, r, name, attrs in _iter_entries(axtree):
        if role is not None and r != role:
            continue
        if name_exact is not None and name != name_exact:
            continue
        if needle is not None and needle not in name.lower():
            continue
        if attrs_contains is not None and attrs_contains not in attrs:
            continue
        if require_visible and "visible" not in attrs:
            continue
        if require_clickable and "clickable" not in attrs:
            continue
        return True
    return False


def _completion_noop() -> str:
    """Visible confirmation pause for states that already satisfy the task."""
    return "noop(500)"


def _dismiss_modal_action(axtree: str) -> str | None:
    """Dismiss modal states observed in live OpenApps AXTree trajectories."""
    if _has_entry(axtree, role="heading", name_exact="Enter Folder Name"):
        bid = _find_exact_bid(axtree, "button", "Cancel", require_clickable=True)
        if bid is not None:
            return f"click('{bid}')"

    if _has_entry(axtree, role="heading", name_exact="Error"):
        bid = _find_exact_bid(axtree, "button", "Close", require_clickable=True)
        if bid is not None:
            return f"click('{bid}')"

    return None


# ---------------------------------------------------------------------------
# Page-state detection
# ---------------------------------------------------------------------------


def _is_on_welcome(axtree: str) -> bool:
    return "Welcome to OpenApps" in axtree or _has_home_launchers(axtree)


def _has_home_launchers(axtree: str) -> bool:
    """True when the scrolled home page shows multiple app-launch cards."""
    visible_launchers = 0
    for app_name in (
        "OpenTodos",
        "OpenCalendar",
        "OpenMessages",
        "OpenMaps",
        "OpenCodeEditor",
    ):
        if _has_entry(axtree, role="link", name_contains=app_name):
            visible_launchers += 1
    return visible_launchers >= 3


def _current_heading(axtree: str) -> str | None:
    """First top-level OpenApps heading on the page."""
    for _, _, role, name, _ in _iter_entries(axtree):
        if role == "heading" and name in {
            "OpenTodos", "OpenCalendar", "OpenMessages",
            "OpenMaps", "OpenCodeEditor",
        }:
            return name
    return None


def _is_calendar_view(axtree: str) -> bool:
    """True on the monthly Calendar grid, not Agenda or detail pages."""
    return any(
        role == "columnheader" and name == "Mon"
        for _, _, role, name, _ in _iter_entries(axtree)
    )


def _detect_app(axtree: str) -> str | None:
    """Infer the current OpenApps app from visible landmarks.

    Random-prefix recovery often lands on pages where the app heading or
    footer is offscreen, so classification must use local controls too.
    """
    if _is_on_welcome(axtree):
        return "Home"

    heading = _current_heading(axtree)
    if heading is not None:
        return heading

    # Calendar subpages and scrolled agenda/calendar views.
    if (
        _parse_calendar_heading(axtree) is not None
        or _in_add_event_dialog(axtree)
        or _has_entry(axtree, role="button", name_exact="Delete Event")
        or _has_entry(axtree, role="link", name_exact="Back to Calendar")
        or _has_entry(axtree, role="button", name_exact="Back to Calendar")
        or _has_entry(axtree, role="button", name_exact="Agenda")
        or _has_entry(axtree, role="button", name_exact="Calendar")
        or _has_entry(axtree, role="button", name_exact="Add Event")
    ):
        return "OpenCalendar"

    if (
        _has_entry(axtree, role="textbox", name_contains="New Todo")
        or _has_entry(axtree, role="button", name_exact="Add")
        or _has_entry(axtree, role="button", name_exact="Remove")
        or _has_entry(axtree, role="button", name_exact="Edit")
        or _has_entry(axtree, role="checkbox")
    ):
        return "OpenTodos"

    if (
        _has_entry(axtree, role="textbox", name_contains="Type a message")
        or _has_entry(axtree, role="textbox", name_contains="Search messages")
        or "chat-bubble" in axtree
    ):
        return "OpenMessages"

    if (
        _has_entry(axtree, role="textbox", name_contains="Search location")
        or _has_entry(axtree, role="button", name_contains="Save location:")
        or _has_entry(axtree, role="button", name_exact="Search")
        or "Saved Locations" in axtree
        or "Current Location Info" in axtree
    ):
        return "OpenMaps"

    if (
        _has_entry(axtree, role="link", name_contains="Code Editor Index Page")
        or _has_entry(axtree, role="button", name_contains="Code Editor Index Page")
        or _has_entry(axtree, role="textbox", name_contains="File name")
        or _has_entry(axtree, name_contains="No file selected")
        or _has_entry(axtree, name_contains="Language:")
        or _has_entry(axtree, name_contains="Theme:")
        or _has_entry(axtree, name_contains="File Explorer")
    ):
        return "OpenCodeEditor"

    return None


def _goto_app(axtree: str, app_name: str) -> str | None:
    """Action to navigate from the welcome page to ``app_name``."""
    # Welcome page links are named e.g. 'OpenTodos OpenTodos' (duplicated).
    bid = _find_bid(axtree, "link", app_name, require_clickable=True)
    if bid is None:
        return None
    return f"click('{bid}')"


def _goto_welcome(axtree: str) -> str | None:
    """Click the "Return to List of Apps" control on any app page.

    Each OpenApps app exposes a ``Return to List of Apps`` element with
    ``href="/"``.  Most apps render it with ``role="button"``; the code
    editor uses a plain anchor (``role="link"``).  Try both.
    """
    for role in ("button", "link"):
        bid = _find_bid(
            axtree, role, "Return to List of Apps", require_clickable=True,
        )
        if bid is not None:
            return f"click('{bid}')"
    return None


def _messages_chat_back_action(axtree: str) -> str | None:
    """Click the visible chat-page back link that returns to the chat list."""
    textbox_idx: int | None = None
    entries = list(_iter_entries(axtree))
    for idx, _, role, name, _ in entries:
        if role == "textbox" and "Type a message" in name:
            textbox_idx = idx
            break
    if textbox_idx is None:
        return None

    fallback_bid: str | None = None
    for idx, bid, role, name, attrs in entries:
        if idx >= textbox_idx:
            break
        if role not in {"link", "button"}:
            continue
        if "visible" not in attrs or "clickable" not in attrs:
            continue
        lname = name.lower()
        if name == "" or "back" in lname or "arrow-left" in lname:
            return f"click('{bid}')"
        if fallback_bid is None:
            fallback_bid = bid
    if fallback_bid is not None:
        return f"click('{fallback_bid}')"
    return None


def _calendar_local_back_action(axtree: str) -> str | None:
    """Return from Calendar detail/add-event subpages to the main calendar."""
    for role in ("button", "link"):
        bid = _find_exact_bid(
            axtree, role, "Back to Calendar", require_clickable=True,
        )
        if bid is not None:
            return f"click('{bid}')"
    return None


def _recover_to_welcome(axtree: str, current_app: str | None) -> str:
    """Use only visible app UI and scrolling to make the home link reachable."""
    action = _goto_welcome(axtree)
    if action is not None:
        return action

    if current_app == "OpenMessages":
        action = _messages_chat_back_action(axtree)
        if action is not None:
            return action
        return "scroll(0, 500)"

    if current_app == "OpenCalendar":
        action = _calendar_local_back_action(axtree)
        if action is not None:
            return action
        return "scroll(0, 500)"

    if current_app == "OpenMaps":
        # The maps return button lives near the top of the sidebar.
        return "scroll(0, -500)"

    # Todos and CodeEditor both expose the return control near the footer.
    return "scroll(0, 500)"


def _ensure_on_target_app(axtree: str, target_app: str) -> str | None:
    """Return a navigation action when we need to (re)reach ``target_app``.

    Recovery is visible-UI faithful: click app launchers/return/back controls
    when they are visible, otherwise scroll until the needed control appears.
    """
    if _is_on_welcome(axtree):
        action = _goto_app(axtree, target_app)
        if action is None:
            return "scroll(0, -500)"
        return action

    current_app = _detect_app(axtree)
    if current_app == target_app:
        return None
    if current_app == "Home":
        action = _goto_app(axtree, target_app)
        if action is not None:
            return action
    if current_app is not None:
        return _recover_to_welcome(axtree, current_app)

    # Unknown but not welcome: prefer a visible app return control, then scroll.
    action = _goto_welcome(axtree)
    if action is not None:
        return action
    return "scroll(0, -500)"


class PolicyStuckError(RuntimeError):
    """Raised when a scripted policy cannot make progress.

    The trajectory runner aborts the affected rollout instead of looping
    on filler noops; the message names the specific stuck condition.
    """


def _stuck(reason: str) -> str:
    """Abort the current rollout with ``reason``.

    Replaces the old ``noop(500)`` fallback — a returned noop would let
    the policy waste ``max_steps`` silently, hiding real bugs.  The
    return type is ``str`` only so call sites read as ``return _stuck(...)``;
    the function never actually returns.
    """
    raise PolicyStuckError(reason)


# ---------------------------------------------------------------------------
# Per-task scripts
# ---------------------------------------------------------------------------


def _textbox_has_value(axtree: str, textbox_name_contains: str, value: str) -> bool:
    """True if a textbox whose name contains ``textbox_name_contains`` has ``value``.

    Handles both quote styles BrowserGym emits for the ``value=`` attribute
    (single quotes are used when the value contains no apostrophes; double
    quotes when the value contains apostrophes — e.g. "Let's meet").
    """
    needle_lower = textbox_name_contains.lower()
    single = f"value='{value}'"
    double = f'value="{value}"'
    for _, _, role, name, attrs in _iter_entries(axtree):
        if role != "textbox" or needle_lower not in name.lower():
            continue
        if single in attrs or double in attrs:
            return True
    return False


def _add_todo_action(axtree: str, todo_name: str) -> str:
    """Add a new todo with the given name.

    Flow: welcome → OpenTodos → fill textbox → click Add → done.
    Idempotent: if the target already appears as a todo item, emit a
    noop.
    """
    nav = _ensure_on_target_app(axtree, "OpenTodos")
    if nav is not None:
        return nav

    # Idempotent: target already in the list as a todo entry.  A textbox
    # value containing the name does NOT count (not yet submitted).
    if f"StaticText '{todo_name}'" in axtree:
        return _completion_noop()

    textbox_bid = _find_bid(axtree, "textbox", "New Todo")
    if textbox_bid is None:
        # Usually means a random prefix scrolled to the todo-list footer.
        return "scroll(0, -500)"

    if _textbox_has_value(axtree, "New Todo", todo_name):
        add_bid = _find_bid(axtree, "button", "Add", require_clickable=True)
        if add_bid is None:
            return _stuck(
                "'Add' button not clickable after filling the New Todo textbox"
            )
        return f"click('{add_bid}')"

    return f"fill('{textbox_bid}', '{todo_name}')"


def _mark_todo_done_action(axtree: str, todo_name: str) -> str:
    """Toggle the checkbox for an existing todo named ``todo_name``.

    Does NOT assume the OpenTodos heading is still on screen — after
    scrolling down in the todos list, the heading gets pushed out of
    view and the axtree no longer reports it.  We detect that we've
    left the welcome page (good enough) and look for the target todo
    directly.
    """
    nav = _ensure_on_target_app(axtree, "OpenTodos")
    if nav is not None:
        return nav

    # Already done?
    checked_bid = _find_bid_near_static_text(
        axtree, todo_name, "checkbox", extra_filter="checked='true'"
    )
    if checked_bid is not None:
        return _completion_noop()

    unchecked_bid = _find_bid_near_static_text(
        axtree, todo_name, "checkbox", extra_filter="checked='false'"
    )
    if unchecked_bid is None:
        # Target todo is not visible — keep scrolling down.  The list
        # can be long (20+ items) so a single page worth isn't always
        # enough.  No state tracking across steps: we just scroll on
        # every turn until the item comes into view OR max_steps fires.
        if _goto_welcome(axtree) is not None:
            return "scroll(0, -500)"
        return "scroll(0, 500)"
    return f"click('{unchecked_bid}')"


def _send_message_action(axtree: str, *, to: str, message: str) -> str:
    """Open a chat with ``to`` and send ``message``.

    Chat-detection can't rely on "heading == to" because the chats-list
    view also shows each contact's name as a heading on their preview
    card.  We detect "inside a chat" by the presence of the
    ``textbox 'Type a message'`` — it only exists on the per-chat page.
    """
    nav = _ensure_on_target_app(axtree, "OpenMessages")
    if nav is not None:
        return nav

    textbox_bid = _find_bid(axtree, "textbox", "Type a message")
    inside_chat = textbox_bid is not None

    if not inside_chat:
        contact_bid = _find_bid(axtree, "link", to, require_clickable=True)
        if contact_bid is None:
            return "scroll(0, -500)"
        return f"click('{contact_bid}')"

    if not _has_entry(axtree, role="heading", name_exact=to):
        action = _messages_chat_back_action(axtree)
        if action is not None:
            return action
        return _stuck(
            f"inside a chat, but the visible heading is not {to!r} and "
            "no chat-list back control is visible"
        )

    # Inside the target chat.  If our message already appeared, we're done.
    if f"StaticText '{message}'" in axtree:
        return _completion_noop()

    if not _textbox_has_value(axtree, "Type a message", message):
        escaped = message.replace("\\", "\\\\").replace("'", "\\'")
        return f"fill('{textbox_bid}', '{escaped}')"

    send_bid = _find_button_after_textbox(axtree, "Type a message")
    if send_bid is None:
        return _stuck("send button not visible after the message textbox")
    return f"click('{send_bid}')"


def _find_button_after_textbox(axtree: str, textbox_name: str) -> str | None:
    """First button appearing after the named textbox in document order."""
    lines = axtree.splitlines()
    anchor = None
    for idx, line in enumerate(lines):
        if f"textbox '{textbox_name}'" in line:
            anchor = idx
            break
    if anchor is None:
        return None
    for line in lines[anchor + 1 : anchor + 6]:
        m = _AXTREE_LINE.match(line)
        if m and m.group("role") == "button":
            return m.group("bid")
    return None


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _parse_calendar_heading(axtree: str) -> tuple[int, int] | None:
    """Return (year, month) from the calendar's "<Month> YYYY" heading.

    OpenCalendar shows a heading like ``[46] heading 'April 2026'`` on
    both Calendar and Agenda views.  Returns None if no such heading is
    visible (e.g. we're in an event-detail view).
    """
    for _, _, role, name, _ in _iter_entries(axtree):
        if role != "heading":
            continue
        parts = name.strip().split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if parts[0] in _MONTH_NAMES:
            return int(parts[1]), _MONTH_NAMES.index(parts[0]) + 1
    return None


def _compare_year_month(
    current: tuple[int, int], target: tuple[int, int],
) -> int:
    """Return -1 if current is before target, 0 if same, +1 if after."""
    if current < target:
        return -1
    if current > target:
        return +1
    return 0


def _ensure_agenda_view(axtree: str) -> str | None:
    """If we're on the monthly grid (Calendar view), click Agenda.

    Returns the action to run, or None if already in Agenda view.
    Detection: Calendar view contains a ``columnheader 'Mon'`` entry
    (day-name header row); Agenda view does not.
    """
    calendar_view = any(
        role == "columnheader" and name == "Mon"
        for _, _, role, name, _ in _iter_entries(axtree)
    )
    if not calendar_view:
        return None
    agenda_bid = _find_bid(axtree, "button", "Agenda", require_clickable=True)
    if agenda_bid is None:
        return None
    return f"click('{agenda_bid}')"


def _navigate_to_month(axtree: str, target_year: int, target_month: int) -> str | None:
    """Return a "< Prev" or "Next >" click if we're not on the target month.

    The Prev/Next buttons step one month at a time; one click per turn.
    Returns None if already on the target.
    """
    current = _parse_calendar_heading(axtree)
    if current is None:
        return None
    cmp = _compare_year_month(current, (target_year, target_month))
    if cmp == 0:
        return None
    if cmp > 0:
        # We're after the target → go back.
        prev_bid = _find_bid(axtree, "button", "Prev", require_clickable=True)
        if prev_bid is None:
            return None
        return f"click('{prev_bid}')"
    # We're before the target → go forward.
    next_bid = _find_bid(axtree, "button", "Next", require_clickable=True)
    if next_bid is None:
        return None
    return f"click('{next_bid}')"


def _event_in_agenda(axtree: str, title: str) -> str | None:
    """Return the bid of an event link whose name starts with ``title``.

    Agenda view renders each event as a link named
    ``'<title> (<recurring>)'`` — e.g. ``'WACV 2026 Abstract Deadline (Online)'``.
    """
    for _, bid, role, name, attrs in _iter_entries(axtree):
        if role == "link" and "clickable" in attrs and name.startswith(title):
            return bid
    return None


def _in_add_event_dialog(axtree: str) -> bool:
    """Robust to vertical scrolling — any Event textbox OR the Submit
    button is a reliable signal that the Add Event dialog is open.
    The "Create New Event" heading alone isn't enough because it
    disappears from the visible axtree once the user scrolls down.
    """
    for _, _, role, name, _ in _iter_entries(axtree):
        if role == "textbox" and name.startswith("Event "):
            return True
        if role == "button" and name == "Submit":
            return True
    return False


def _in_event_detail_view(axtree: str, title: str) -> bool:
    """True if we're on the event-detail page for ``title`` (has Delete)."""
    if _find_bid(axtree, "button", "Delete Event", require_clickable=True) is None:
        return False
    # The heading shows the event title; confirm it matches.
    return f"heading '{title}'" in axtree


def _in_any_event_detail_view(axtree: str) -> bool:
    """True on a Calendar event-detail page, regardless of event title."""
    return _find_bid(axtree, "button", "Delete Event", require_clickable=True) is not None


def _add_event_action(
    axtree: str,
    *,
    title: str,
    date: str,
    description: str | None = None,
    location: str | None = None,
    url: str | None = None,
    invitees: list[str] | None = None,
) -> str:
    """Create a calendar event with the given fields.

    ``date`` must be ``YYYY-MM-DD``.  None fields are skipped (left empty)
    — this matches the upstream task config where description/location
    are optional and verification ignores fields that the goal doesn't
    specify.  ``invitees`` is passed as a comma-separated string when
    non-empty.
    """
    # Already done?  Look for the event in the agenda.
    year, month = map(int, date.split("-")[:2])
    target_ym = (year, month)
    current_ym = _parse_calendar_heading(axtree)

    nav = _ensure_on_target_app(axtree, "OpenCalendar")
    if nav is not None:
        return nav

    if _in_event_detail_view(axtree, title):
        return _completion_noop()

    if _in_any_event_detail_view(axtree):
        back = _calendar_local_back_action(axtree)
        if back is not None:
            return back
        return "scroll(0, -500)"

    if _in_add_event_dialog(axtree):
        return _fill_add_event_form(
            axtree,
            title=title,
            date=date,
            description=description,
            location=location,
            url=url,
            invitees=invitees,
        )

    # Normal agenda/calendar view.  Ensure we're on Agenda view (Add
    # Event button lives only there).
    ensure_agenda = _ensure_agenda_view(axtree)
    if ensure_agenda is not None:
        return ensure_agenda

    # Already added?
    if current_ym is not None and current_ym == target_ym:
        if _event_in_agenda(axtree, title) is not None:
            return _completion_noop()

    # Navigate to the target month if we need to see existing events
    # for idempotency — but for `add`, we can skip straight to clicking
    # Add Event: the form accepts any date regardless of the visible
    # month.
    add_event_bid = _find_bid(axtree, "button", "Add Event", require_clickable=True)
    if add_event_bid is None:
        return "scroll(0, 500)"
    return f"click('{add_event_bid}')"


def _fill_add_event_form(
    axtree: str,
    *,
    title: str,
    date: str,
    description: str | None,
    location: str | None,
    url: str | None,
    invitees: list[str] | None,
) -> str:
    """Fill a currently-visible field of the Add Event dialog, or submit.

    The form is taller than one viewport, so the policy must scroll to
    reveal later fields.  Stateless logic:

    1. Walk the requested fields top-to-bottom.  For each, check whether
       the textbox appears in the *current* axtree.  If the textbox is
       visible and lacks the target value, fill it — done for this
       turn.
    2. If every requested field whose textbox is visible already has
       its value, check for the Submit button.  If visible, click it.
    3. Otherwise scroll down to reveal more of the form.

    Textboxes that have scrolled out of view are simply skipped — we
    don't know whether they're filled from prior turns, but since the
    UI preserves values, we only need to fill fields that are *both*
    visible *and* empty.  If a required field somehow got missed, the
    Submit click will fail validation, and we'll fall back to scrolling
    (eventually making it visible again).
    """
    invitees_text = ", ".join(invitees) if invitees else None

    requested_fields = [
        ("Event title", title),
        ("Event date", date),
        ("Event description", description),
        ("Event location", location),
        ("Event invitees", invitees_text),
        ("Event URL", url),
    ]

    for textbox_name, value in requested_fields:
        if value is None:
            continue
        bid = _find_bid(axtree, "textbox", textbox_name)
        if bid is None:
            continue   # not visible this turn, may already be filled
        if _textbox_has_value(axtree, textbox_name, value):
            continue
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"fill('{bid}', '{escaped}')"

    submit_bid = _find_bid(axtree, "button", "Submit", require_clickable=True)
    if submit_bid is not None:
        return f"click('{submit_bid}')"
    return "scroll(0, 300)"


def _remove_event_action(
    axtree: str,
    *,
    title: str,
    date: str,
) -> str:
    """Navigate to the event's month, open the event, click Delete.

    ``date`` is ``YYYY-MM-DD`` for the target month selection.
    """
    nav = _ensure_on_target_app(axtree, "OpenCalendar")
    if nav is not None:
        return nav

    if _in_add_event_dialog(axtree):
        action = _goto_welcome(axtree)
        if action is not None:
            return action
        return "scroll(0, 500)"

    # Are we on the event detail page with our event?  Click Delete.
    if _in_event_detail_view(axtree, title):
        del_bid = _find_bid(axtree, "button", "Delete Event", require_clickable=True)
        if del_bid is None:
            return _stuck(
                f"on event-detail page for {title!r} but 'Delete Event' "
                "button is not clickable"
            )
        return f"click('{del_bid}')"

    if _in_any_event_detail_view(axtree):
        back = _calendar_local_back_action(axtree)
        if back is not None:
            return back
        return "scroll(0, -500)"

    # Switch to Agenda view if needed.
    ensure_agenda = _ensure_agenda_view(axtree)
    if ensure_agenda is not None:
        return ensure_agenda

    # Navigate to the target month.
    year, month = map(int, date.split("-")[:2])
    nav = _navigate_to_month(axtree, year, month)
    if nav is not None:
        return nav

    # On target month in agenda view — click the event link.
    event_bid = _event_in_agenda(axtree, title)
    if event_bid is None:
        current = _parse_calendar_heading(axtree)
        if current == (year, month):
            # If the footer controls are visible too, we have reached the end
            # of the target-month agenda and the event is visibly absent.
            if _find_bid(axtree, "button", "Add Event", require_clickable=True):
                return _completion_noop()
            if _goto_welcome(axtree) is not None:
                return _completion_noop()
            return "scroll(0, 500)"
        return "scroll(0, -500)"
    return f"click('{event_bid}')"


def _save_place_action(axtree: str, place_query: str) -> str:
    """Search for a place and click its "Save location: …" button.

    Flow: welcome → OpenMaps → fill search box → click Search → wait for
    results → click "Save location: …".
    """
    nav = _ensure_on_target_app(axtree, "OpenMaps")
    if nav is not None:
        return nav

    # If a "Save location: …" result button is visible, click it.
    save_bid = _find_bid(
        axtree, "button", "Save location:", require_clickable=True,
    )
    if save_bid is not None:
        return f"click('{save_bid}')"

    place_marker = place_query.split(",", 1)[0]
    if (
        _has_entry(axtree, role="button", name_contains=f"Delete {place_marker}")
        or _has_entry(axtree, role="button", name_contains=f"Marker for {place_marker}")
    ):
        return _completion_noop()

    search_bid = _find_bid(axtree, "textbox", "Search location")
    if search_bid is None:
        return "scroll(0, -500)"

    if not _textbox_has_value(axtree, "Search location", place_query):
        escaped = place_query.replace("\\", "\\\\").replace("'", "\\'")
        return f"fill('{search_bid}', '{escaped}')"

    # Textbox is filled but results haven't appeared yet.  Current
    # OpenApps versions ship without a dedicated Search button — the
    # search fires on Enter.  If an explicit Search button happens to
    # exist (older config), click it; otherwise press Enter to submit.
    search_btn = _find_bid(axtree, "button", "Search", require_clickable=True)
    if search_btn is not None:
        return f"click('{search_btn}')"
    return f"press('{search_bid}', 'Enter')"


# ---------------------------------------------------------------------------
# Dispatch + registration
# ---------------------------------------------------------------------------


# Task-specific parameters (name, date, invitees etc) come from the
# upstream OpenApps config so we keep them in sync with ground-truth.
_TASK_PARAMS: dict[str, dict[str, Any]] = {
    # Todos
    "add_call_mom_to_my_todo": {"todo_name": "Call Mom"},
    "mark_water_plants_as_done": {"todo_name": "Water plants"},
    # Messenger
    "message_bob_to_meet": {"to": "Bob", "message": "Let's meet"},
    # Maps
    "save_paris_to_my_favorite_places": {"place_query": "Paris, France"},
    # Calendar — add
    "add_meeting_with_dennis": {
        "title": "Dennis-Bob", "date": "2026-04-01",
    },
    "add_christmas_shopping_event": {
        "title": "Shopping for Christmas gifts", "date": "2025-12-14",
    },
    # NOTE: this task is currently unsolvable through the UI due to an
    # upstream OpenApps mismatch — the Calendar backend stores the
    # ``invitees`` field as a raw string (whatever the textbox contains),
    # but ``AddEventTask.get_target_state`` builds the target with
    # ``invitees=["Einstein"]`` (a list from the yaml config).  No input
    # format tried (plain text, JSON array, comma-separated) produces the
    # expected stored list.  The scripted policy still emits the correct
    # UI action sequence, so once the upstream serialization is fixed (or
    # ``AddEventTask`` is adjusted to store lists on submit) the policy
    # will succeed without further changes.  DeepSeek text-only and
    # Gemma 4 VLM also scored 0/5 on this task in pass@k runs.
    "add_paper_reading_meeting_with_einstein": {
        "title": "Einstein-Bob", "date": "2027-04-01",
        "description": "paper reading", "location": "New York City",
        "invitees": ["Einstein"],
    },
    # Calendar — remove
    "remove_wacv_abstract_deadline": {
        "title": "WACV 2026 Abstract Deadline", "date": "2025-07-11",
    },
}


# Goal-substring markers used to infer task_name when the policy is
# registered under the umbrella ``open_apps`` name (no static task in
# metadata).  The OpenApps adapter sets the initial user prompt to
# ``"Goal: {goal}\n\n{axtree}"``, so the conversation's first user
# message always carries the goal line.  Each substring here is unique
# to exactly one task in upstream ``config/tasks/all_tasks.yaml``.
_GOAL_MARKERS: tuple[tuple[str, str], ...] = (
    ("Dennis", "add_meeting_with_dennis"),
    ("Christmas shopping", "add_christmas_shopping_event"),
    ("Einstein", "add_paper_reading_meeting_with_einstein"),
    ("WACV", "remove_wacv_abstract_deadline"),
    ("Call Mom", "add_call_mom_to_my_todo"),
    ("Water plants", "mark_water_plants_as_done"),
    ("Bob", "message_bob_to_meet"),
    ("Paris", "save_paris_to_my_favorite_places"),
)


def _infer_task_name_from_messages(messages: list[Any]) -> str | None:
    """Walk the chat history's first user message for a goal marker.

    Returns the matching ``OPEN_APPS_TASKS`` name, or ``None`` if no
    marker is found.  Used by the umbrella ``open_apps`` policy when
    task_indices cycle through all 8 tasks in a single collection run.
    """
    first_user: str | None = None
    for message in messages:
        content = getattr(message, "content", None)
        if getattr(message, "role", None) == "user" and content:
            first_user = content
            break
    if not first_user:
        return None
    for marker, task_name in _GOAL_MARKERS:
        if marker in first_user:
            return task_name
    return None


def _scripted_action(state_text: str, metadata: dict[str, Any]) -> str:
    """Dispatch to the per-task script.

    Resolves ``task_name`` in priority order:
      1. ``metadata['task_name']`` if set — the per-task policies
         registered as ``open_apps_<task>`` always carry this.
      2. Otherwise scan the first user message in
         ``metadata['_messages']`` (injected by ``ScriptedBackend``)
         for a goal marker unique to one task.  Used by the single
         umbrella ``open_apps`` policy, which is what you want when the
         collection cycles all 8 tasks via ``task_indices``.
    """
    modal_action = _dismiss_modal_action(state_text)
    if modal_action is not None:
        return modal_action

    task_name = metadata.get("task_name")
    if task_name is None:
        messages = metadata.get("_messages") or []
        task_name = _infer_task_name_from_messages(messages)
        if task_name is None:
            return _stuck(
                "umbrella open_apps policy: no task_name in metadata and "
                "no goal marker matched in the first user message"
            )
    params = _TASK_PARAMS.get(task_name)
    if params is None:
        return _stuck(
            f"no _TASK_PARAMS entry for inferred task_name {task_name!r}"
        )

    if task_name == "add_call_mom_to_my_todo":
        return _add_todo_action(state_text, **params)
    if task_name == "mark_water_plants_as_done":
        return _mark_todo_done_action(state_text, **params)
    if task_name == "message_bob_to_meet":
        return _send_message_action(state_text, **params)
    if task_name == "save_paris_to_my_favorite_places":
        return _save_place_action(state_text, **params)
    if task_name in (
        "add_meeting_with_dennis",
        "add_christmas_shopping_event",
        "add_paper_reading_meeting_with_einstein",
    ):
        return _add_event_action(state_text, **params)
    if task_name == "remove_wacv_abstract_deadline":
        return _remove_event_action(state_text, **params)

    return _stuck(f"no dispatcher branch for task_name {task_name!r}")


def _available_actions(state_text: str, _metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return the currently-valid action set for stochastic wrapping.

    OpenApps rewards compare the whole application state against the
    reset-time target.  A random prefix that clicks destructive controls
    (Remove, Delete Event, Submit, checkbox toggles, Send, Save
    location, etc.) can corrupt unrelated app state in ways the AXTree
    policy cannot reconstruct.  Keep stochastic prefixes faithful to the
    recovery problem we care about: visible navigation, local subpages,
    modal openings/closings, scrolling, and noops.
    """
    actions: list[str] = ["noop(500)", "scroll(0, 500)", "scroll(0, -500)"]
    for _, bid, role, name, attrs in _iter_entries(state_text):
        if "clickable" in attrs:
            if _is_safe_stochastic_click(role, name):
                actions.append(f"click('{bid}')")
    return tuple(actions)


def _is_safe_stochastic_click(role: str, name: str) -> bool:
    """Whether epsilon-random sampling may click this visible element."""
    if role == "link":
        return True
    if role != "button":
        return False

    safe_exact = {
        "Return to List of Apps",
        "Back to Calendar",
        "Agenda",
        "Calendar",
        "Add Event",
        "< Prev",
        "Next >",
        "Cancel",
        "Close",
        "Zoom in",
        "Zoom out",
        "New Folder",
    }
    if name in safe_exact:
        return True
    return name.startswith("Marker for ")


for _task_name in _TASK_PARAMS:
    register_policy(
        f"open_apps_{_task_name}",
        _scripted_action,
        metadata={
            "task_name": _task_name,
            # Callable action space: evaluated per-turn against the
            # current accessibility tree so stochastic wrapping samples
            # from valid bids rather than a stale fixed list.
            "actions": _available_actions,
        },
    )


# Umbrella policy: when task_indices cycle across all 8 tasks in a
# single run, a single backend dispatches per-trajectory by parsing
# the initial goal from the conversation history.  Prefer this over
# the per-task ``open_apps_<task>`` policies for multi-task
# collection configs.
register_policy(
    "open_apps",
    _scripted_action,
    metadata={
        # No ``task_name`` here — it's inferred per-call from the goal
        # in the first user message.
        "actions": _available_actions,
    },
)


__all__ = ["_scripted_action"]
