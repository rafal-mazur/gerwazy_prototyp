import serial
from typing import Any


class SerialPort(serial.Serial):
    def send(self, *messages: Any) -> None:
        for message in messages:
            self.write(bytes(str(message), 'utf-8'))
            self.write(b'#')
    
    def read_msg(self, stop: str ='#') -> str:
        return str(self.read_until(bytes(stop, 'utf-8'))).strip(stop)
