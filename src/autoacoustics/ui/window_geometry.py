"""Conservative desktop viewport bounds, including a decorated window frame."""
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication,QScrollArea,QSizePolicy

def available_geometry(window):
    screen=window.screen() or QApplication.primaryScreen()
    return screen.availableGeometry() if screen else None

def fit_to_available_screen(window,preferred=None,margin=12):
    bounds=available_geometry(window)
    if bounds is None:return
    size=QSize(*preferred) if isinstance(preferred,tuple) else preferred or window.size()
    # Before show, frame metrics can be zero. Reserve enough for a title bar;
    # showEvent calls this again once the actual frame has been created.
    frame_width=max(16,window.frameGeometry().width()-window.width())
    frame_height=max(48,window.frameGeometry().height()-window.height())
    width=max(1,min(size.width(),bounds.width()-2*margin-frame_width))
    height=max(1,min(size.height(),bounds.height()-2*margin-frame_height))
    window.resize(width,height)
    window.move(bounds.left()+max(margin,(bounds.width()-window.width()-frame_width)//2),
                bounds.top()+max(margin,(bounds.height()-window.height()-frame_height)//2))

def scroll_page(content,name=''):
    scroll=QScrollArea();scroll.setObjectName(name);scroll.setWidgetResizable(True);scroll.setWidget(content)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
    return scroll

def flexible_wrapped_label(label):
    label.setWordWrap(True);label.setMinimumWidth(0)
    policy=label.sizePolicy();policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored);label.setSizePolicy(policy)
