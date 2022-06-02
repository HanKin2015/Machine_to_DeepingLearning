import win32com.client

wmi = win32com.client.GetObject ("winmgmts:")
for usb in wmi.InstancesOf ("Win32_USBHub"):
    print(usb.DeviceID)
   
for usb in wmi.InstancesOf ("win32_usbcontrollerdevice"):
    print(usb.Dependent)