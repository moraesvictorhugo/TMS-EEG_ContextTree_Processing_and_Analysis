import time
import threading
from Components.remote_control import RemoteControl
from Components import constants as consts

class UpdateNavigationInfo:

    def __init__(self, number_of_rc):
        '''
        Define information for updating navigation information; characteristics of the object
        '''

        self.number_of_rc = number_of_rc
        self.rc = [None] * self.number_of_rc
        self.target_status = [None] * self.number_of_rc
        self.marker_label = [None] * self.number_of_rc
        self.all_target_status = None
        self.thread = None
        self.status_lock = threading.Lock()

    def connect(self, address, ports):
        for i in range(len(self.rc)):
            self.rc[i] = RemoteControl('http://' + address + ':' + str(ports[i])) # Refers to the first pulse, the Conditioning Stimulus (CS)
            self.rc[i].try_connect()
        self.call_thread()

    def processes(self):
        while True:
            for i in range(len(self.rc)):
                self.target_status[i], self.marker_label[i] = self.get_buffer_msg(self.rc[i], self.target_status[i], self.marker_label[i])
            self.all_target_status = all(self.target_status)

    def get_buffer_msg(self, rc, target, marker):
        '''
        Check the message status in the Buffer that comes from the relay_server
        '''

        buffer = rc.get_buffer()

        if len(buffer) == 0:
            pass

        for i in range(len(buffer)):
            if buffer[i]['topic'] == consts.PUB_MESSAGES[0]:
                target = buffer[i]["data"]["state"]
            elif buffer[i]['topic'] == consts.PUB_MESSAGES[1]:
                marker = buffer[i]["data"]["name"]

        time.sleep(0.1)
        return target, marker

    def send_trigger_to_navigation(self):
        if consts.create_navigation_marker:
            topic = 'Create marker'
            data = {}
            for rc in self.rc:
                if rc is not None:
                    rc.send_message(topic, data)

    def call_thread(self):
        '''
        Call the processes function as a thread
        '''

        self.thread = threading.Thread(target=self.processes, daemon=True)
        self.thread.start()


    #TODO: stop thread
    # def stop_thread(self):

# delete_marker = False
# unset_marker = False
#

#
# def send_to_navigation_delete_marker(rc):
#     global delete_marker
#     if delete_marker:
#         topic = 'Delete marker'
#         data = {'Enabled':'True'}
#         rc.send_message(topic, data)
# def send_to_navigation_unset_marker(rc):
#     global unset_marker
#     if unset_marker:
#         topic = 'Unset marker'
#         data = {'Enabled':'False'}
#         rc.send_message(topic, data)