import pyxinput
import random
import time


class XboxControllerEmulator:
    """
    Emulates a xbox 360 controller using pyxinput

    You need to install pyxinput, see: https://github.com/bayangan1991/PYXInput
        In case something happens to the repository here is a fork
        with the exact version I am using for this project: https://github.com/ikergarcia1996/PYXInput

    This is how I installed ScpVBus in Windows 10 (required to emulate the controller):
    1) Run PowerShell as administrator
    2) Enable PowerShell scrips: Set-ExecutionPolicy unrestricted
    3) Enable testing mode (required to install an unsigned driver): bcdedit /set testsigning on
    4) Disable integrity checks (required to install an unsigned driver): bcdedit /set nointegritychecks off
    5) Reboot
    6) Run cmd as administrator
    7) Execute the PYXInput/pyxinput/ScpVBus-x64/install.bat
    8) Go back to default windows config:   bcdedit /set nointegritychecks on
                                            bcdedit /set testsigning off
                                            set-executionpolicy remotesigned
    """

    virtual_controller: pyxinput.vController

    def __init__(self):
        self.virtual_controller = pyxinput.vController()
        print("Virtual xbox 360 controller crated")

    def stop(self):
        self.virtual_controller.UnPlug()
        print("Virtual xbox 360 controller removed")

    def set_axis_lx(self, lx: float):
        """
        Sets the x value for the right stick

        Input:
         -lx: float in range [-1,1]
        Output:
        """

        assert -1.0 <= lx <= 1.0, f"Controller values must be in range [-1,1]. x: {lx}"
        self.virtual_controller.set_value("AxisLx", lx)

    def set_axis_ly(self, ly: float):
        """
        Sets the x value for the left stick

        Input:
         -ly: float in range [-1,1]
        Output:
        """

        assert -1.0 <= ly <= 1.0, f"Controller values must be in range [-1,1]. y: {ly}"
        self.virtual_controller.set_value("AxisLy", ly)

    def set_axis(self, lx: float, ly: float):
        """
        Sets the x and y values for the right stick

        Input:
         -lx: float in range [-1,1]
         -ly: float in range [-1,1]
        Output:
        """

        self.set_axis_lx(lx)
        self.set_axis_ly(ly)

    def set_trigger_lt(self, lt: float):
        """
        Sets the t value for the left trigger

        Input:
         -lt: float in range [-1,1]
        Output:
        """

        assert -1.0 <= lt <= 1.0, f"Controller values must be in range [-1,1]. lt: {lt}"
        self.virtual_controller.set_value("TriggerL", lt)

    def set_trigger_rt(self, rt: float):
        """
        Sets the t value for the right trigger

        Input:
         -rt: float in range [-1,1]
        Output:
        """

        assert -1.0 <= rt <= 1.0, f"Controller values must be in range [-1,1]. rt: {rt}"
        self.virtual_controller.set_value("TriggerR", rt)

    def set_controller_state(self, lx: float, lt: float, rt: float):
        """
        Sets the x value for the left stick and the t value for the left and right triggers

        Input:
         -lx: float in range [-1,1]
         -lt: float in range [-1,1]
         -rt: float in range [-1,1]
        Output:
        """

        self.set_axis_lx(lx)
        self.set_trigger_lt(lt)
        self.set_trigger_rt(rt)

    def test(self, num_tests: int = 10, delay: float = 0.5):
        """
        Tests if the virtual controller works correctly using random values

        Input:
         -num_tests: Integer, Number of test to perform (Iterations of random values)
         -delay: Float, number of seconds to wait between each test
        Output:
        """
        print("Testing left stick...")
        for test_no in range(num_tests):
            lx, ly = (
                1 - (random.random() * 2),
                1 - (random.random() * 2),
            )
            print(f"LX: {lx} \t LY: {ly}")
            self.set_axis(
                lx=lx,
                ly=ly,
            )
            time.sleep(delay)
        self.set_axis(
            lx=0.0,
            ly=0.0,
        )

        print("Testing left trigger...")
        for test_no in range(num_tests):
            lt = 1 - (random.random() * 2)
            print(f"LT: {lt}")
            self.set_trigger_lt(lt=lt)
            time.sleep(delay)
        self.set_trigger_lt(lt=0.0)

        print("Testing right trigger...")
        for test_no in range(num_tests):
            rt = 1 - (random.random() * 2)
            print(f"RT: {rt}")
            self.set_trigger_rt(rt=rt)
            time.sleep(delay)
        self.set_trigger_rt(rt=0.0)
