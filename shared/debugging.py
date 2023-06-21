import curses
import time
import threading

class LoadingAnimation:
    def __init__(self, message: str = ''):
        self.is_running = True
        self.screen = curses.initscr()
        self.message = message
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        
    def start(self):
        self.thread = threading.Thread(target=self.run_animation)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()

    def run_animation(self):
        i = 0
        while self.is_running:
            self.screen.clear()
            self.screen.addstr(0, 0, self.message + '.' * (i % 4), curses.color_pair(1))
            self.screen.refresh()
            time.sleep(1)
            i += 1

        self.screen.clear()
        self.screen.addstr(0, 0, 'Training model' + '.' * i, curses.color_pair(2))
        self.screen.refresh()
        time.sleep(2)
        curses.endwin()
        
    def set_message(self, message: str):
        self.message = message


#Usage
loading = LoadingAnimation()
loading.start()

# simulate a delay
time.sleep(10)

loading.stop()