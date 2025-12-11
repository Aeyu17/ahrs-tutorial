import h5py
from dataclasses import dataclass
import numpy as np

@dataclass
class SensorData:
    freq: float
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray | None
    quat: np.ndarray


def importADPM(file: str, id: str) -> SensorData:
    with h5py.File(file) as f:
        data = f[id]

        calibrated_sensor_data = data['Calibrated']

        freq = float(data.attrs['SampleRate'])
        print(f'Sample Rate: {freq}')

        accel = calibrated_sensor_data['Accelerometers'][:]
        gyro = calibrated_sensor_data['Gyroscopes'][:]
        if data.attrs['MagnetometersEnabled'] == 0:
            mag = calibrated_sensor_data['Magnetometers'][:]
        else:
            mag = None
        
        quat = calibrated_sensor_data['Orientation'][:]

        return SensorData(freq=freq, accel=accel, gyro=gyro, mag=mag, quat=quat)