from screeninfo import get_monitors
import cv2
import os


class slmDisplay:
    def __init__(self, screen=1, flag_open_device=False, flag_window_open=False, h_thread_handle=None,
                 flag_thread_closed=None, flag_update=None, flag_thread_working=None):
        self.__flag_open_device = flag_open_device
        self.__flag_window_open = flag_window_open
        self.__flag_thread_closed = flag_thread_closed
        self.__flag_update = flag_update
        self.__flag_thread_working = flag_thread_working
        self.__h_thread_handle = h_thread_handle
        self.geometry = get_monitors()[screen]
        self.window = 'slmDisplay'

    def open(self):
        if not self.__flag_window_open:
            cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(self.window, self.geometry.x, self.geometry.y)
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(1)
        pass

    def update(self, filePath):
        # print('using update process')
        if os.path.isfile(filePath):
            img = cv2.imread(filePath)
        else:
            img = filePath
        cv2.imshow(self.window, img)
        cv2.waitKey(1)
        pass

    def close(self):
        cv2.destroyWindow(self.window)
        pass

# if __name__ == '__main__':
#     filepath = 'Img_001.jpg'
#     slm = slmDisplay()
#     slm.open()
#     slm.update(filepath)
#     slm.close()
