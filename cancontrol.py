import can
from threading import Thread

class CanControl():
    car_angle = 0
    def send_one(id, data):
        with can.Bus(interface="socketcan", channel="can0", bitrate=500000) as bus:
            
            msg = can.Message(
                arbitration_id=id, data=data, is_extended_id=True
            )
            try:
                bus.send(msg)
            
            except can.CanError:
                print("Message NOT sent")

    def receive(self):    
        while True:
            with can.Bus(interface="socketcan", channel="can0", bitrate=500000) as bus:                       
                msg = bus.recv(0.01)
                if msg is not None and msg.arbitration_id==0x00010000: 
                    pos = msg.data[0]
                    self.car_angle = int.from_bytes(msg.data[1:3], "big")
                    if pos ==1:
                        self.car_angle = -abs(self.car_angle)

    def send_angle_2_car(self, angle):
        pos = 1 if angle >0 else 0
        data = pos.to_bytes(1, 'big')+abs(angle).to_bytes(2, 'big')
        self.send_one(id=0x01000003, data=[1,0])
        
        self.send_one(id=0x01010100, data=data)    
        self.send_one(id=0x01010007, data=[1])

    def get_car_angle(self):
        return self.car_angle

    def run(self):
        thread = Thread(target = self.receive)
        thread.start()
