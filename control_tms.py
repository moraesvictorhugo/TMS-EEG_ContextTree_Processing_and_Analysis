import serial.tools.list_ports
import magicpy as mp
import subprocess
import unittest
import time


class TestMagicPy(unittest.TestCase):
    ser_port = '/dev/ttyS0'   # /dev/ttyUSB0
    device_port = './ttydevice'
    client_port = './ttyclient'
    try:
        ports = serial.tools.list_ports.comports()
        ser_port, _, _ = ports[0]
    except IndexError:
        print("Trying to emulate serial port.")
        cmd = ['socat', '-d', '-d', f'PTY,link={device_port},raw,echo=0', f'PTY,link={client_port},raw,echo=0']
        ser_port = device_port
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1)

    def test_init(self):
        stimulator = mp.MagVenture(self.ser_port)  # Initialize the stimulator object
        stimulator.connect()

    def test_arm(self):
        stimulator = mp.MagVenture(self.ser_port)
        stimulator.connect()
        assert stimulator.arm() == (None, None)

    def test_set_current_dir(self):
        # stimulator.connect()
        stimulator = mp.MagVenture(self.ser_port, wait_s=0, wait_l=0)
        with self.assertRaises(ValueError):
            stimulator.set_current_dir('Reversed')

    def test_set_waveform(self):
        stimulator = mp.MagVenture(self.ser_port, wait_s=0, wait_l=0)
        with self.assertRaises(ValueError):
            assert stimulator.set_waveform('Biphasic') == (None, None)

    def test_set_amplitude(self):
        stimulator = mp.MagVenture(self.ser_port)
        stimulator.connect()
        stimulator.arm()  # Enable stimulator

        assert stimulator.set_amplitude(50, get_response=False) == (None, None)

    def test_list_ports(self):
        mp.list_serial_ports()

    def test_init_parallel(self):
        with self.assertRaises(FileNotFoundError) as _:
            mp.MagVenture(self.ser_port, ttl_port='0')

    def test_dec2hex_padded(self):
        stimulator = mp.MagVenture(self.ser_port)
        assert stimulator.dec2hex_padded(0) == '00'
        assert stimulator.dec2hex_padded(10) == '0a'
        with self.assertRaises(TypeError):
            stimulator.dec2hex_padded('0')

    def test_dec2bin_padded(self):
        stimulator = mp.MagVenture(self.ser_port)
        assert stimulator.dec2bin_padded(0) == '00000000'
        assert stimulator.dec2bin_padded(10) == '00001010'
        with self.assertRaises(TypeError):
            stimulator.dec2bin_padded('0')


if __name__ == '__main__':
    unittest.main()
