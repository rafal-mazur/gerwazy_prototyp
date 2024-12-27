import serial
import time
from typing import Any


class Message:
    def __init__(self, content: Any):
        self.content = str(content)
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}(\'{self.content}\')'
    
    def __type__(self) -> str:
        return f'<class \'{__class__.__name__}\'>'

    
class SerialPort(serial.Serial):
    def send(self, *messages: Message) -> None:
        if len(messages) == 0:
            return
        
        for message in messages:
            self.write(bytes(message.content, 'utf-8'))
            self.write(b'#')
    
    def read_msg(self, stop: str ='#') -> Message:
        return Message(str(self.read_until(bytes(stop, 'utf-8'))).strip(stop))


if __name__ == '__main__':
    with SerialPort('/dev/serial0') as port:
        licznik = 0
        while True:
            time.sleep(0.2)
            licznik += 1
            port.send(Message(licznik))