import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations, WindowFunctions
from phue import Bridge

# Adam Luchjenbroers https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def main(board, args):

    b = Bridge('192.168.1.9')

    b.connect()

    board_descr = board.get_board_descr(args.board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    board.start_stream()

    for _ in range(20):
        # board.start_stream () # use this for default options
        time.sleep(2)
        # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
        data = board.get_board_data()  # get all data and remove it from internal buffer

        eeg_channels = board_descr['eeg_channels']
        # second eeg channel of synthetic board is a sine wave at 10Hz, should see huge alpha
        eeg_channel = eeg_channels[1]
        # optional detrend
        DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                    WindowFunctions.BLACKMAN_HARRIS.value)

        band_power_theta = DataFilter.get_band_power(psd, 3.0, 6.0)
        band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
        band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
        print("theta/alpha/beta:%f", band_power_theta, band_power_alpha , band_power_beta)

        value = band_power_theta + band_power_alpha + band_power_beta
        print('1',value)
        value = int(translate(value, 0, 200, 0, 65535))
        print('2',value)
        b.set_light(4, 'hue', value)

    board.stop_stream()

def setup():
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file



    board = BoardShim(args.board_id, params)
    board.prepare_session()

    main(board, args)

    board.release_session()




if __name__ == "__main__":
    setup()