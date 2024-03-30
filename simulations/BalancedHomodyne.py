import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as constants
import csv

class BalancedHomodyne:
    """
    A class to simulate balanced homodyne detection for optical signals, providing methods to generate coherent pulses,
    apply various modifications and effects to these pulses, and perform detection using balanced homodyne technique.

    Attributes:
        central_wavelength (float): Central wavelength of the pulses in meters.
        average_power_lo (float): Average power of the local oscillator (LO) in Watts.
        average_power_signal (float): Average power of the signal in Watts.
        repetition_rate (float): Repetition rate of the pulses in Hz.
        temporal_width_lo (float): Temporal width of the LO pulse in seconds.
        temporal_width_signal (float): Temporal width of the signal pulse in seconds.
        cross_area (float): Cross-sectional area of the beam in square meters.
        time_window (float): Time window for simulation in seconds.
        grid_points (int): Number of grid points in the simulation time window.
    """

    def __init__(self, 
                 central_wavelength=1550e-9, # Central wavelength in meters
                 average_power_lo=1e-6, # Average power of LO in Watts
                 average_power_signal=1e-7, # Average power of the signal in Watts
                 repetition_rate=61e6, # Repetition rate in Hz
                 temporal_width_lo=300e-15, # Temporal width of LO pulse in seconds
                 temporal_width_signal=300e-15, # Temporal width of signal pulse in seconds
                 cross_area=np.pi*1e-6, # Cross-sectional area of the beam in square meters
                 time_window=6000e-15, # Time window for simulation in seconds
                 grid_points=5000): # Number of grid points in the time window
        
        # Initialize attributes with provided parameters
        self.central_wavelength = central_wavelength
        self.average_power_lo = average_power_lo
        self.average_power_signal = average_power_signal
        self.repetition_rate = repetition_rate
        self.temporal_width_lo = temporal_width_lo
        self.temporal_width_signal = temporal_width_signal
        self.cross_area = cross_area
        self.time_window = time_window
        self.grid_points = grid_points

        # Create a time grid for the simulation, centered around zero
        self.time_grid = np.linspace(-self.time_window/2, self.time_window/2, self.grid_points)

    def calculate_photon_statistics(self, average_power):
        """
        Calculate and print photon statistics including energy per photon, energy per pulse, average photons per pulse,
        and photons per second, based on the average power of the pulse.

        Parameters:
            average_power (float): The average power of the pulse for which to calculate statistics.

        Returns:
            tuple: A tuple containing energy per photon, energy per pulse, average photons per pulse, and photons per second.
        """
        # Calculate energy per photon using Planck's constant and speed of light
        energy_per_photon = constants.h * constants.c / self.central_wavelength
        print(f"Energy per photon: {energy_per_photon}")

        # Calculate energy per pulse based on the repetition rate
        energy_per_pulse = average_power / self.repetition_rate
        print(f"Energy per pulse: {energy_per_pulse}")

        # Calculate average photons per pulse
        average_photons_per_pulse = energy_per_pulse / energy_per_photon
        print(f"Average photons per pulse: {average_photons_per_pulse}")

        # Calculate photons per second
        photons_per_second = average_photons_per_pulse * self.repetition_rate
        print(f"Photons per second: {photons_per_second}")

        return energy_per_photon, energy_per_pulse, average_photons_per_pulse, photons_per_second

    def calculate_field_amplitude(self, average_power, temporal_width):
        """
        Calculate the field amplitude of a pulse based on its average power and temporal width.

        Parameters:
            average_power (float): The average power of the pulse.
            temporal_width (float): The temporal width of the pulse.

        Returns:
            float: The calculated field amplitude of the pulse.
        """
        # Calculate energy per pulse and peak power
        energy_per_pulse = average_power / self.repetition_rate
        peak_power = (2 * energy_per_pulse) / (temporal_width * np.sqrt(np.pi / np.log(2)))

        # Calculate field amplitude using peak power and beam cross-sectional area
        field_amplitude = np.sqrt((2 * peak_power) / (constants.c * constants.epsilon_0 * self.cross_area))

        return field_amplitude

    def calculate_phase_shift(self, opd):
        """
        Calculate the phase shift of a pulse based on the optical path difference (OPD).

        Parameters:
            opd (float): The optical path difference in meters.

        Returns:
            float: The calculated phase shift of the pulse.
        """
        # Calculate phase shift using the central wavelength and OPD
        phase_shift = (2 * np.pi / self.central_wavelength) * opd
        return phase_shift

    def distort_pulse_spectrum(self, t, pulse, bandwidth, distortion_function):
        """
        Apply spectral distortion to a pulse based on a specified distortion function.

        Parameters:
            t (np.ndarray): The time grid array.
            pulse (np.ndarray): The pulse to be distorted.
            bandwidth (float): The bandwidth of the distortion.
            distortion_function (function): The function used to apply the distortion.

        Returns:
            np.ndarray: The distorted pulse.
        """
        # Calculate central angular frequency and bandwidth in angular frequency
        central_omega = 2 * np.pi * constants.c / self.central_wavelength
        bandwidth_omega = 2 * np.pi * constants.c * bandwidth / (self.central_wavelength ** 2 - bandwidth ** 2 / 4)

        # Apply FFT to the pulse, create a frequency array, and apply the distortion function
        E_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pulse)))
        freqs = np.fft.fftshift(np.fft.fftfreq(t.size, d=t[1] - t[0]))
        omega = 2 * np.pi * freqs
        E_w_distorted = E_w * distortion_function(omega, central_omega, bandwidth_omega)

        # Apply inverse FFT to get the distorted pulse in time domain
        pulse_distorted = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(E_w_distorted)))

        return pulse_distorted

    def random_distortion(self, omega, central_omega, bandwidth_omega, phase_variance=1, amplitude_variance=0.1):
        """
        Generate a random distortion factor for a given range of angular frequencies.

        Parameters:
            omega (np.ndarray): Array of angular frequencies.
            central_omega (float): Central angular frequency.
            bandwidth_omega (float): Bandwidth in angular frequency.
            phase_variance (float, optional): Variance of the phase distortion. Default is 1.
            amplitude_variance (float, optional): Variance of the amplitude distortion. Default is 0.1.

        Returns:
            np.ndarray: The random distortion factor for each frequency.
        """
        # Initialize distortion factor as an array of ones
        distortion_factor = np.ones_like(omega, dtype=complex)

        # Determine frequency bounds within which distortion is applied
        lower_bound = central_omega - bandwidth_omega / 2
        upper_bound = central_omega + bandwidth_omega / 2
        within_bandwidth = (omega >= lower_bound) & (omega <= upper_bound)

        # Apply random phase and amplitude distortion within the specified bandwidth
        distortion_factor[within_bandwidth] *= (
            (1 + np.random.normal(0, amplitude_variance, np.sum(within_bandwidth))) *
            np.exp(1j * np.random.normal(0, phase_variance, np.sum(within_bandwidth)))
        )

        return distortion_factor

    def add_quantum_noise(self, pulse, delay, time_window, average_photons_per_pulse):
        """
        Add quantum noise to a pulse within a specified time window centered around a delay.

        Parameters:
            pulse (np.ndarray): The pulse to which noise is to be added.
            delay (float): The center of the time window where noise is added, in seconds.
            time_window (float): The duration of the time window where noise is added, in seconds.
            average_photons_per_pulse (float): The average number of photons per pulse, used to scale the noise.

        Returns:
            np.ndarray: The pulse with added quantum noise.
        """
        # Copy the pulse to avoid modifying the original
        pulse_with_noise = np.copy(pulse)

        # Find indices within the specified time window
        noise_indices = np.where(np.abs(self.time_grid - delay) <= time_window / 2)[0]

        # Calculate mode volume and quantum noise level
        omega = 2 * np.pi * constants.c / self.central_wavelength
        mode_volume = self.cross_area * self.central_wavelength
        nois_amplitude = np.sqrt(average_photons_per_pulse)
        quantum_noise = np.random.uniform(-nois_amplitude/2, nois_amplitude/2, len(pulse_with_noise[noise_indices]))

        # Add quantum noise within the specified time window
        pulse_with_noise[noise_indices] += (1 + quantum_noise) * np.exp(1j * quantum_noise)

        return pulse_with_noise

    def apply_gdd_to_pulse(self, pulse, gdd_fs2):
        """
        Apply group delay dispersion (GDD) to a pulse.

        Parameters:
            pulse (np.ndarray): The pulse to which GDD is to be applied.
            gdd_fs2 (float): The amount of GDD in femtoseconds squared.

        Returns:
            np.ndarray: The pulse with applied GDD.
        """
        # Calculate central angular frequency
        omega_0 = 2 * np.pi * constants.c / self.central_wavelength

        # Apply FFT to the pulse and create a frequency array
        E_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pulse)))
        freqs = np.fft.fftshift(np.fft.fftfreq(self.time_grid.size, d=self.time_grid[1] - self.time_grid[0]))
        omega = 2 * np.pi * freqs

        # Calculate phase shift due to GDD and apply it to the pulse in frequency domain
        phase_shift = 0.5 * gdd_fs2 * 1e-30 * (omega - omega_0) ** 2
        E_w_gdd = E_w * np.exp(1j * phase_shift)

        # Apply inverse FFT to get the pulse with applied GDD in time domain
        E_t_gdd = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(E_w_gdd)))

        return E_t_gdd

    def coherent_pulse(self, field, delay=0, gdd_fs2=0, distortion=False, loss_per_s=0, offset_loss=0, distortion_bandwidth=0):
        """
        Generate a coherent pulse with optional modifications such as GDD, spectral distortion, and loss.

        Parameters:
            field (str): Specifies which field ('signal' or 'lo') to generate the pulse for.
            delay (float, optional): Delay to apply to the pulse in seconds. Default is 0.
            gdd_fs2 (float, optional): Amount of group delay dispersion to apply in femtoseconds squared. Default is 0.
            distortion (bool, optional): Whether to apply spectral distortion to the pulse. Default is False.
            loss_per_s (float, optional): Loss per second to apply to the pulse. Default is 0.
            offset_loss (float, optional): Offset for the loss calculation. Default is 0.
            distortion_bandwidth (float, optional): Bandwidth for the spectral distortion. Default is 0.

        Returns:
            np.ndarray: The generated coherent pulse with applied modifications.
        """
        # Determine average power and temporal width based on specified field
        if(field == 'signal'):
            average_power = self.average_power_signal
            temporal_width = self.temporal_width_signal
        else:
            average_power = self.average_power_lo
            temporal_width = self.temporal_width_lo

        # Calculate photon statistics for the pulse
        _, _, average_photons_per_pulse, _ = self.calculate_photon_statistics(average_power)

        # Calculate central frequency and field amplitude
        central_frequency = constants.c / self.central_wavelength
        amplitude = self.calculate_field_amplitude(average_power, temporal_width)
        print(f"Field amplitude = {amplitude} V/m")

        # Apply loss to the amplitude based on delay and offset
        loss_factor = 1 - (loss_per_s * np.abs(delay-offset_loss)/(2*np.pi))
        amplitude *= loss_factor

        # Initialize pulse array and generate pulse centered around the specified delay
        pulse = np.zeros_like(self.time_grid, dtype=complex)
        pulse_center = delay
        time_from_center = self.time_grid - pulse_center
        pulse += amplitude * np.exp(-((time_from_center) ** 2) / (2 * temporal_width**2)) * np.exp(1j * (2 * np.pi * central_frequency * time_from_center))
        
        # Apply GDD to the pulse
        pulse = np.copy(self.apply_gdd_to_pulse(pulse, gdd_fs2*1e30))

        # Optionally apply spectral distortion to the pulse
        if(distortion):
            pulse = np.copy(self.distort_pulse_spectrum(self.time_grid, pulse, self.central_wavelength, distortion_bandwidth, self.random_distortion))

        # Add quantum noise to the pulse
        pulse = np.copy(self.add_quantum_noise(pulse, delay, 5*temporal_width, average_photons_per_pulse))

        return pulse

    def balanced_homodyne_detection(self, signal_pulse, lo_pulse, electric_noise_level=0.2):
        """
        Perform balanced homodyne detection on a signal pulse using a local oscillator (LO) pulse.

        Parameters:
            signal_pulse (np.ndarray): The signal pulse to be detected.
            lo_pulse (np.ndarray): The local oscillator pulse used for detection.
            electric_noise_level (float, optional): The level of electric noise to add to the detection output. Default is 0.2.

        Returns:
            np.ndarray: The output of the balanced homodyne detection.
        """
        # Combine signal and LO pulses to get the output at two ports
        output_port1 = (signal_pulse + lo_pulse) / np.sqrt(2)
        output_port2 = (signal_pulse - lo_pulse) / np.sqrt(2)

        # Calculate detector output as the difference in intensity between the two ports
        detector_output = np.abs(output_port1 * np.conj(output_port1)) - np.abs(output_port2 * np.conj(output_port2))

        # Add random electric noise to the detector output
        electric_noise = np.random.normal(-electric_noise_level / 2, electric_noise_level / 2, signal_pulse.shape)
        detector_output += electric_noise

        return detector_output
    
    def perform_multiple_measurements(self, N, delay=0, file_path="data/waveform_data_"):
        """
        Perform N measurements, generating a new signal and LO pulse for each measurement,
        and computing the detector signal. Returns concatenated detector signals and a
        corresponding time array.

        Parameters:
            N (int): The number of measurements to perform.
            delay (float, optional): Delay to apply to the pulse in seconds. Default is 0.
            file_path (str): Path to the CSV file where the data will be saved.

        Returns:
            tuple: Two arrays, the first concatenating all the detector signal measurements,
                and the second being the corresponding time array adjusted for N measurements.
        """
        # Initialize an empty array for concatenated detector signals
        concatenated_detector_signals = np.array([])

        # For each measurement
        for i in range(N):
            # Generate a new signal and LO pulse
            signal_pulse = self.coherent_pulse('signal')
            lo_pulse = self.coherent_pulse('lo', delay=delay)

            # Generate the detector signal for the current pair of signal and LO pulses
            detector_signal = self.balanced_homodyne_detection(signal_pulse, lo_pulse)

            # Concatenate the current detector signal to the cumulative array
            concatenated_detector_signals = np.concatenate((concatenated_detector_signals, detector_signal))

        # Generate the corresponding time array
        # Note: time_grid is centered around 0, but we want to start from 0 for the first measurement
        # and extend to N * time_window for the last measurement.
        measurement_duration = self.time_grid[-1] - self.time_grid[0]  # Duration of one measurement
        concatenated_time_array = np.linspace(0, N * measurement_duration, len(concatenated_detector_signals))

        # Open the file in write mode
        file_path = file_path + str(delay*constants.c) + ".csv"
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the header
            csvwriter.writerow(['date', '\'30 MAR 2024\''])
            csvwriter.writerow(['time', '\'14:51:40:25\''])
            csvwriter.writerow(['Time (SECOND)', 'Voltage (VOLT)'])

            # Write the time and detector values
            for time, signal in zip(concatenated_time_array, concatenated_detector_signals):
                csvwriter.writerow([time, signal])

        return concatenated_detector_signals, concatenated_time_array
    
    def plot_pulses(self, signal_pulse, lo_pulse, detector_output):
        """
        Plot the signal and LO pulses along with the detector output from the balanced homodyne detection.

        Parameters:
            signal_pulse (np.ndarray): The signal pulse to be plotted.
            lo_pulse (np.ndarray): The local oscillator pulse to be plotted.
            detector_output (np.ndarray): The detector output from the balanced homodyne detection to be plotted.
        """
        # Plotting the intensities of signal and LO pulses on the same figure with dual y-axes
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()  # Get the current Axes instance
        line1, = ax1.plot(self.time_grid, np.abs(signal_pulse)**2, label='Intensity of Chirped Signal Pulse', color='blue')
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Intensity of Chirped Signal Pulse', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2 = ax1.twinx()
        line2, = ax2.plot(self.time_grid, np.abs(lo_pulse)**2, label='Intensity of LO Pulse', linestyle='--', color='green')
        ax2.set_ylabel('Intensity of LO Pulse', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title('Coherent State Pulses')

        # Plotting the detector output
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_grid, detector_output, label='Balanced Homodyne Detection Output')
        plt.title('Balanced Homodyne Detection Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Detector Output')
        plt.grid(True)
        plt.legend()
        plt.xlim(self.time_grid[0], self.time_grid[-1])
        plt.ylim(-np.max(lo_pulse)**2, np.max(lo_pulse)**2)
        plt.show()

    def animate_pulses(self, optical_path_difference):
        """
        Create an animation of the signal and LO pulses, and the balanced homodyne detection output as the optical
        path difference (OPD) is varied.

        Parameters:
            optical_path_difference (np.ndarray): Array of OPD values to simulate over the animation.
        """
        # Initialize the figure and axes for the animation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        line1, = ax1.plot([], [], label='Signal Pulse')
        line2, = ax1.plot([], [], label='LO Pulse', linestyle='--')
        line3, = ax2.plot([], [], label='Balanced Homodyne Output')
        ax1.set_xlim(self.time_grid[0], self.time_grid[-1])
        ax1.set_ylim(0, 5e10)
        ax1.set_title('Coherent State Pulses')
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Intensity')
        ax1.legend()
        ax1.grid(True)
        ax2.set_xlim(self.time_grid[0], self.time_grid[-1])
        ax2.set_ylim(-3e10, 3e10)
        ax2.set_title('Balanced Homodyne Detection Simulation')
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Detector Output')
        ax2.grid(True)
        ax2.legend()

        # Initialize function for the animation
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            return line1, line2, line3

        # Define the animation function
        def animate(i):
            opd = optical_path_difference[i]
            delay = opd / constants.c  # Convert OPD to delay in seconds

            # Generate signal and LO pulses with the current phase shift applied to the LO
            signal_pulse = self.coherent_pulse('signal')
            lo_pulse = self.coherent_pulse('lo', delay=delay)
            detector_output = self.balanced_homodyne_detection(signal_pulse, lo_pulse)

            # Update the plots with the new pulses and detector output
            line1.set_data(self.time_grid, np.abs(signal_pulse)**2)
            line2.set_data(self.time_grid, np.abs(lo_pulse)**2)
            line3.set_data(self.time_grid, detector_output)

            # Update OPD text on the plot
            ax1.text(0.05, 0.95, f'OPD: {opd*1e6:.2f} µm', transform=ax1.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            return line1, line2, line3

        # Create and save the animation
        ani = FuncAnimation(fig, animate, frames=len(optical_path_difference), init_func=init, blit=True, interval=50)
        ani.save('balanced_homodyne_detection.gif', writer='pillow', fps=120)

        plt.show()
