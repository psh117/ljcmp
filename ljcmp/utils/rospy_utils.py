import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarkerFeedback, InteractiveMarker, InteractiveMarkerControl


def interactive_marker_control_6dof():
    controls = []
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "rotate_x"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "rotate_y"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    return controls

def interactive_marker_control_4dof():
    controls = []
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    controls.append(control)

    return controls