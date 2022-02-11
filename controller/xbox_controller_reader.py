import pygame
import time
import sys
import logging


class XboxControllerReader:
    """
    Reads the current state of a Xbox Controller
    May work with other similar controllers too

    You need to install pygame to use this class: https://www.pygame.org/wiki/GettingStarted
    """

    joystick: pygame.joystick
    name: str
    joystick_id: int

    def __init__(self, total_wait_secs: int = 10):
        """
        Init

        - Total_wait_secs: Integer, number of seconds to wait. Pygame take some time to initialize, during the first
        seconds you may get wrong readings for the controller, waiting a few seconds before starting reading
        is recommended.
        """
        pygame.init()
        pygame.joystick.init()
        try:
            self.joystick = pygame.joystick.Joystick(0)
        except pygame.error:
            logging.warning(
                "No controller found, ensure that your controlled is connected and is recognized by windows"
            )
            sys.exit()

        self.joystick.init()
        self.name = self.joystick.get_name()
        self.joystick_id = self.joystick.get_id()

        for delay in range(int(total_wait_secs), 0, -1):
            print(
                f"Initializing controller reader, waiting {delay} seconds to prevent wrong readings...",
                end="\r",
            )
            time.sleep(1)

        print(f"Recording input from: {self.name} ({self.joystick_id})\n")

    def read(self) -> (float, float, float, float):
        """
        Reads the current state of the controller

        Input:

        Output:
         -lx: Float, current X value of the right stick in range [-1,1]
         -lt: Float, current L value in range [-1,1]
         -rt: Float, current R value in range [-1,1]
        """
        _ = pygame.event.get()
        lx, lt, rt = (
            self.joystick.get_axis(0),
            self.joystick.get_axis(4),
            self.joystick.get_axis(5),
        )

        return lx, lt, rt
