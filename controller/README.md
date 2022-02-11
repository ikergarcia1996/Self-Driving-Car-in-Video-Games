# Xbox controller emulator setup (PYXInput)
You don't need to follow this guide if you intend to use the keyboard as the TEDD1104 control mode. PYXInput allows the creation of a virtual xbox controller that TEDD1104 can use for driving.

PYXInput: https://github.com/bayangan1991/PYXInput  
* In case something happens to the repository here is a fork with the exact version I am using for this project: https://github.com/ikergarcia1996/PYXInput

## Install ScPVBUS
First you need to install ScPVBus. Tested in Windows 10 and Windows 11
0) Clone the PYXInput repository
1) Run PowerShell as administrator
2) Enable PowerShell scrips: 
``` 
Set-ExecutionPolicy unrestricted 
```
3) Enable testing mode (required to install an unsigned driver):
```
bcdedit /set testsigning on
```
4) Disable integrity checks (required to install an unsigned driver):
```
bcdedit /set nointegritychecks off
```
6) Reboot
7) Run PowerShell as administrator
8) Execute the PYXInput/pyxinput/ScpVBus-x64/install.bat
```
cd path-where-you-cloned-pyxinput\PYXInput-master\pyxinput\ScpVBus-x64
.\install.bat
```

9) Go back to default Windows config (Will take effect after reboot):   

```
bcdedit /set nointegritychecks on
bcdedit /set testsigning off
set-executionpolicy remotesigned
```


## Install PYXInput
```
pip install PYXInput
```
## TEST PYXINPUT
Run python
```
import pyxinput
pyxinput.test_virtual()
```

Expected output

```
>>> import pyxinput
>>> pyxinput.test_virtual()
Testings multiple connections
Connecting Controller:
Available: [2, 3, 4]
This ID: 1
Connecting Controller:
Available: [3, 4]
This ID: 2
Connecting Controller:
Available: [4]
This ID: 3
Connecting Controller:
Available: []
This ID: 4
Connecting Controller:
Done, disconnecting controllers.
Available: [1, 2, 3, 4]
Testing Value setting
Connecting Controller:
This ID: 1
Available: [2, 3, 4]
Setting TriggerR and AxisLx:
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Done, disconnecting controller.
Available: [1, 2, 3, 4]
```


