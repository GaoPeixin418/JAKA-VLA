"""Offline contact acceptance for the archive-rig grasp dataset."""

def lab_forbidden_contact(pair):
    """Allow intended finger contact and the fixed base resting on the table."""
    arm = [name for name in pair if name.startswith(('left:', 'right:'))]
    if not arm:
        return False
    if len(arm) == 2:
        return True
    other = next(name for name in pair if name not in arm)
    if arm[0].endswith(':Link_0') and other == 'scene:table':
        return False
    return not (other == 'scene:box' and arm[0] in (
        'left:finger_left', 'left:finger_right', 'left:finger_third'))
