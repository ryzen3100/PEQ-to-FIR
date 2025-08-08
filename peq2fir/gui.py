import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore[import-untyped]
import os
import threading
import numpy as np
from .output_handler import OutputHandler, WAVOutputHandler, TextOutputHandler, JSONOutputHandler
import json
from .converter import PEQtoFIR
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Set matplotlib to use DejaVu Sans which has better minus sign support
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False  # Use ASCII minus instead of Unicode

class PEQtoFIRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PEQ to FIR Converter")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.sample_rates = [44100, 48000]
        self.num_taps = tk.IntVar(value=4095)
        self.phase_type = tk.StringVar(value="linear")
        self.bit_depth = tk.IntVar(value=16)
        self.channels = tk.IntVar(value=1)  # Number of channels (1=mono, 2=stereo)
        self.apply_file_preamp = tk.BooleanVar(value=True)  # Apply preamp from file
        self.apply_auto_preamp = tk.BooleanVar(value=True)  # Apply automatic clipping prevention
        self.peq_filters = []
        self.loaded_filename = None  # Track loaded filename
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Text editor and controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(0, 5))
        
        # Title
        title_label = ttk.Label(left_panel, text="PEQ Settings Editor", font=('Helvetica', 12, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Text editor with drag & drop
        editor_frame = ttk.LabelFrame(left_panel, text="PEQ Configuration (Drag & Drop .txt files here)")
        editor_frame.grid(row=1, column=0, sticky='nsew', pady=(0, 10))
        left_panel.rowconfigure(1, weight=1)
        
        self.text_editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD, width=50, height=20,
                                                     font=('Consolas', 10))
        self.text_editor.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)
        
        # Enable drag and drop on the entire editor frame
        editor_frame.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        editor_frame.dnd_bind('<<Drop>>', self.drop_file)  # type: ignore[attr-defined]
        
        # Also enable on the text editor itself
        self.text_editor.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
        self.text_editor.dnd_bind('<<Drop>>', self.drop_file)  # type: ignore[attr-defined]
        
        # Add default example
        self.text_editor.insert(tk.END, """# AutoEQ Style PEQ Settings
# Format: Filter N: ON TYPE Fc FREQ Hz Gain GAIN dB Q VALUE
# TYPE: PK (Peaking), HS (High Shelf), LS (Low Shelf)
# Optional: Preamp: VALUE dB

Preamp: -3.7 dB
Filter 1: ON PK Fc 65 Hz Gain -4.5 dB Q 0.8
Filter 2: ON PK Fc 150 Hz Gain 3.0 dB Q 1.2
Filter 3: ON PK Fc 800 Hz Gain -2.0 dB Q 2.0
Filter 4: ON PK Fc 3000 Hz Gain 4.5 dB Q 3.0
Filter 5: ON PK Fc 6000 Hz Gain -3.5 dB Q 4.0
Filter 6: ON HS Fc 10000 Hz Gain 2.0 dB Q 0.707

# You can also use simple format:
# peaking 100 1.41 -3.0
# highshelf 10000 0.707 3.0""")
        
        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(button_frame, text="Load File", command=self.load_file).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_editor).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Parse PEQ", command=self.parse_peq).grid(row=0, column=2, padx=5)
        
        # Right panel - Settings and visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, rowspan=2, sticky='nsew')
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(right_panel, text="Conversion Settings")
        settings_frame.grid(row=0, column=0, sticky='we', pady=(0, 10))
        
        # Settings grid
        row = 0
        ttk.Label(settings_frame, text="Filter Taps:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        taps_frame = ttk.Frame(settings_frame)
        taps_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        for taps in [2047, 4095, 8191]:
            ttk.Radiobutton(taps_frame, text=str(taps), variable=self.num_taps, 
                           value=taps).pack(side=tk.LEFT, padx=5)
        
        row += 1
        ttk.Label(settings_frame, text="Phase Type:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        phase_frame = ttk.Frame(settings_frame)
        phase_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(phase_frame, text="Linear", variable=self.phase_type, 
                       value="linear").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(phase_frame, text="Minimum", variable=self.phase_type, 
                       value="minimum").pack(side=tk.LEFT, padx=5)
        
        row += 1
        ttk.Label(settings_frame, text="Bit Depth:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        bit_frame = ttk.Frame(settings_frame)
        bit_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(bit_frame, text="16-bit", variable=self.bit_depth, 
                       value=16).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(bit_frame, text="24-bit", variable=self.bit_depth, 
                       value=24).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(bit_frame, text="32-bit float", variable=self.bit_depth, 
                       value=32).pack(side=tk.LEFT, padx=5)
        
        row += 1
        ttk.Label(settings_frame, text="Channels:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        channels_frame = ttk.Frame(settings_frame)
        channels_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(channels_frame, text="Mono", variable=self.channels, 
                       value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(channels_frame, text="Stereo", variable=self.channels, 
                       value=2).pack(side=tk.LEFT, padx=5)
        row += 1
        preamp_frame = ttk.Frame(settings_frame)
        preamp_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Checkbutton(preamp_frame, text="Apply File Preamp", 
                       variable=self.apply_file_preamp).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Checkbutton(preamp_frame, text="Apply Auto Preamp (Clipping Prevention)", 
                       variable=self.apply_auto_preamp).pack(side=tk.LEFT)
        
        row += 1
        ttk.Label(settings_frame, text="Sample Rates: 44.1 kHz, 48 kHz (both will be generated)",
                 font=('Helvetica', 9, 'italic')).grid(row=row, column=0, columnspan=2, 
                                                       sticky=tk.W, padx=5, pady=2)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(right_panel, text="Frequency Response Preview")
        viz_frame.grid(row=1, column=0, sticky='nsew')
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure with two subplots
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121)  # Left plot for target
        self.ax2 = self.fig.add_subplot(122)  # Right plot for verification
        self.fig.subplots_adjust(left=0.08, right=0.98, wspace=0.25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Bottom panel - Progress and controls
        bottom_panel = ttk.Frame(main_frame)
        bottom_panel.grid(row=2, column=0, columnspan=2, sticky='we', pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_panel, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, padx=10, pady=5)
        
        self.status_label = ttk.Label(bottom_panel, text="Ready")
        self.status_label.grid(row=0, column=1, padx=10, pady=5)
        
        # Buttons frame
        button_row = ttk.Frame(bottom_panel)
        button_row.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Convert button
        self.convert_button = ttk.Button(button_row, text="Convert to FIR", 
                                       command=self.convert_to_fir,
                                       style='Accent.TButton')
        self.convert_button.grid(row=0, column=0, padx=5)
        
        # Verify button
        self.verify_button = ttk.Button(button_row, text="Verify FIR File", 
                                      command=self.verify_fir_file)
        self.verify_button.grid(row=0, column=1, padx=5)
        
    def drop_file(self, event):
        # Handle different formats of file paths
        try:
            # Try to parse the dropped data
            files = self.root.tk.splitlist(event.data)
        except:
            # If splitlist fails, try direct string processing
            files = event.data.strip()
            # Remove curly braces if present (Windows)
            if files.startswith('{') and files.endswith('}'):
                files = files[1:-1]
            files = [files]
        
        if files and files[0]:
            file_path = files[0].strip()
            # Handle file:// URLs
            if file_path.startswith('file://'):
                file_path = file_path[7:]
            # Handle Windows paths
            file_path = os.path.normpath(file_path)
            
            if os.path.exists(file_path) and file_path.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.text_editor.delete('1.0', tk.END)
                    self.text_editor.insert('1.0', content)
                    self.loaded_filename = os.path.basename(file_path).replace('.txt', '')
                    self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                    print(f"Successfully loaded: {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                    print(f"Error loading file: {e}")
            else:
                messagebox.showwarning("Warning", "Please drop a valid .txt file")
                print(f"Invalid file: {file_path}")
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select PEQ configuration file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                self.text_editor.delete('1.0', tk.END)
                self.text_editor.insert('1.0', content)
                self.loaded_filename = os.path.basename(file_path).replace('.txt', '')
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def clear_editor(self):
        self.text_editor.delete('1.0', tk.END)
        self.peq_filters = []
        self.loaded_filename = None
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()
        self.status_label.config(text="Editor cleared")
    
    def parse_peq(self):
        content = self.text_editor.get('1.0', tk.END)
        self.peq_filters = []
        preamp_value = None
        
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse Preamp line
            if line.startswith('Preamp:'):
                try:
                    parts = line.split()
                    preamp_value = float(parts[1])
                    continue
                except:
                    continue
            
            # Parse AutoEQ format
            if line.startswith('Filter'):
                parts = line.split()
                if len(parts) >= 9 and parts[2] == 'ON':
                    try:
                        filt = {
                            'type': parts[3].lower(),
                            'freq': float(parts[5]),
                            'gain': float(parts[8]),
                            'q': float(parts[11]) if len(parts) > 11 else 0.707
                        }
                        # Map types
                        if filt['type'] == 'pk':
                            filt['type'] = 'peaking'
                        elif filt['type'] == 'hs':
                            filt['type'] = 'highshelf'
                        elif filt['type'] == 'ls':
                            filt['type'] = 'lowshelf'
                        self.peq_filters.append(filt)
                    except:
                        continue
            
            # Parse simple format
            else:
                parts = line.split()
                if len(parts) == 4:
                    try:
                        filt = {
                            'type': parts[0].lower(),
                            'freq': float(parts[1]),
                            'q': float(parts[2]),
                            'gain': float(parts[3])
                        }
                        self.peq_filters.append(filt)
                    except:
                        continue
        
        if self.peq_filters:
            status_text = f"Parsed {len(self.peq_filters)} filters"
            if preamp_value is not None:
                status_text += f" (Preamp: {preamp_value} dB)"
                # Add preamp as a special filter for tracking
                self.peq_filters.insert(0, {'type': 'preamp', 'gain': preamp_value})
            self.status_label.config(text=status_text)
            self.preview_response()
        else:
            messagebox.showwarning("Warning", "No valid PEQ filters found in the text")
    
    def preview_response(self):
        if not self.peq_filters:
            return
        
        self.ax1.clear()
        self.ax2.clear()
        
        converter = PEQtoFIR(fs=48000, num_taps=self.num_taps.get())
        
        # Use the unified method to get final target response
        frequencies, response_db = converter.get_final_target_response(
            self.peq_filters,
            use_file_preamp=self.apply_file_preamp.get(),
            use_auto_preamp=self.apply_auto_preamp.get()
        )
        
        # Plot target response on left
        self.ax1.semilogx(frequencies, response_db, 'b-', linewidth=2)
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_ylabel('Magnitude (dB)')
        self.ax1.set_title('Target PEQ Response')
        self.ax1.grid(True, which='both', alpha=0.3)
        self.ax1.set_xlim(20, 24000)
        y_min = np.min(response_db) - 3
        y_max = np.max(response_db) + 3
        self.ax1.set_ylim(y_min, y_max)
        
        # Clear right plot
        self.ax2.set_title('FIR Verification (Load a FIR file)')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude (dB)')
        self.ax2.grid(True, which='both', alpha=0.3)
        
        self.canvas.draw()
    
    def convert_to_fir(self):
        if not self.peq_filters:
            self.parse_peq()
            if not self.peq_filters:
                return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return
        
        # Disable button during conversion
        self.convert_button.config(state='disabled')
        self.progress_var.set(0)
        
        # Run conversion in thread
        thread = threading.Thread(target=self.perform_conversion, args=(output_dir,))
        thread.start()
    
    def perform_conversion(self, output_dir):
        try:
            import datetime
            
            total_steps = len(self.sample_rates) * 2  # For each sample rate: convert + save
            current_step = 0
            
            results = {}
            
            # Extract file preamp value ONCE (for metadata only)
            file_preamp_db = next((f['gain'] for f in self.peq_filters if f.get('type') == 'preamp'), 0.0)
            
            # Get preamp settings from checkboxes ONCE
            use_file_preamp = self.apply_file_preamp.get()
            use_auto_preamp = self.apply_auto_preamp.get()
            
            print(f"GUI: Found preamp in file: {file_preamp_db} dB")
            print(f"GUI: File preamp checkbox: {use_file_preamp}")
            print(f"GUI: Auto preamp checkbox: {use_auto_preamp}")
            
            # Create output handlers
            wav_handler = WAVOutputHandler(bit_depth=self.bit_depth.get())
            text_handler = TextOutputHandler()
            json_handler = JSONOutputHandler()
            
            for fs in self.sample_rates:
                self.root.after(0, lambda fs=fs: self.status_label.config(
                              text=f"Converting for {fs} Hz..."))
                
                converter = PEQtoFIR(fs=fs, num_taps=self.num_taps.get())
                
                # Design FIR filter with the unified method
                fir_coeffs = converter.design_fir_filter(
                    self.peq_filters,
                    use_file_preamp=use_file_preamp,
                    use_auto_preamp=use_auto_preamp,
                    phase_type=self.phase_type.get()
                )
                
                current_step += 1
                progress = (current_step / total_steps) * 100
                self.root.after(0, self.progress_var.set, progress)
                
                # Generate filename
                channel_str = "Stereo_" if self.channels.get() == 2 else ""
                if self.loaded_filename:
                    # Use loaded filename
                    base_name = f"{self.loaded_filename}_FIR_{channel_str}{self.phase_type.get().capitalize()}_{self.num_taps.get()}taps_{fs}Hz"
                else:
                    # Use timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    base_name = f"{timestamp}_FIR_{self.phase_type.get().capitalize()}_{self.num_taps.get()}taps_{fs}Hz"
                
                # Prepare metadata
                metadata = {
                    'fs': fs,
                    'num_taps': self.num_taps.get(),
                    'phase_type': self.phase_type.get(),
                    'num_channels': self.channels.get(),
                    'basename': base_name
                }
                
                # Save files using handlers
                wav_handler.save(fir_coeffs, metadata, output_dir)
                text_handler.save(fir_coeffs, metadata, output_dir)
                
                # Analyze results using the EXACT same settings
                analysis = converter.analyze_filter(
                    fir_coeffs, 
                    self.peq_filters,
                    use_file_preamp=use_file_preamp,
                    use_auto_preamp=use_auto_preamp
                )
                results[fs] = {
                    'latency_ms': analysis['latency_ms'],
                    'max_error_db': analysis['max_error_db'],
                    'rms_error_db': analysis['rms_error_db']
                }
                
                current_step += 1
                progress = (current_step / total_steps) * 100
                self.root.after(0, self.progress_var.set, progress)
            
            # Prepare metadata for JSON handler
            metadata = {
                'peq_filters': self.peq_filters,
                'num_taps': self.num_taps.get(),
                'phase_type': self.phase_type.get(),
                'bit_depth': self.bit_depth.get(),
                'num_channels': self.channels.get(),
                'file_preamp_applied': self.apply_file_preamp.get(),
                'auto_preamp_applied': self.apply_auto_preamp.get(),
                'file_preamp_value': file_preamp_db if self.apply_file_preamp.get() else 0.0,
                'results': results
            }
            
            # Save metadata using JSON handler
            json_handler.save(np.array([]), metadata, output_dir)
            
            # Show completion message
            msg = "Conversion completed!\n\n"
            for fs, res in results.items():
                msg += f"{fs} Hz:\n"
                msg += f"  - Latency: {res['latency_ms']:.1f} ms\n"
                msg += f"  - Max error: {res['max_error_db']:.2f} dB\n"
                msg += f"  - RMS error: {res['rms_error_db']:.2f} dB\n\n"
            msg += f"Files saved to: {output_dir}"
            
            self.root.after(0, lambda: messagebox.showinfo("Success", msg))
            self.root.after(0, lambda: self.status_label.config(text="Conversion completed"))
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="Conversion failed"))
        finally:
            self.root.after(0, lambda: self.convert_button.config(state='normal'))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def verify_fir_file(self):
        """Load and verify a FIR filter file"""
        file_path = filedialog.askopenfilename(
            title="Select FIR filter file to verify",
            filetypes=[("WAV files", "*.wav"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.wav'):
                # Load WAV file
                fs, data = wavfile.read(file_path)
                if len(data.shape) > 1:
                    # Take first channel if stereo
                    fir_coeffs = data[:, 0]
                else:
                    fir_coeffs = data
                
                # Convert to float
                if data.dtype == np.int16:
                    fir_coeffs = fir_coeffs.astype(np.float64) / 32768.0
                elif data.dtype == np.int32:
                    fir_coeffs = fir_coeffs.astype(np.float64) / 2147483648.0
                else:
                    fir_coeffs = fir_coeffs.astype(np.float64)
                    
            elif file_path.endswith('.txt'):
                # Load text file
                fir_coeffs = np.loadtxt(file_path)
                # Assume 48kHz if loading from text
                fs = 48000
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            # Calculate frequency response
            from scipy.signal import freqz  # type: ignore[import-untyped]
            w, h = freqz(fir_coeffs, worN=8192, fs=fs)
            magnitude_db = 20 * np.log10(np.abs(h))
            
            # Update right plot
            self.ax2.clear()
            self.ax2.semilogx(w, magnitude_db, 'r-', linewidth=2)
            self.ax2.set_xlabel('Frequency (Hz)')
            self.ax2.set_ylabel('Magnitude (dB)')
            self.ax2.set_title(f'FIR Verification: {os.path.basename(file_path)}')
            self.ax2.grid(True, which='both', alpha=0.3)
            
            # Match the scale with left plot
            if hasattr(self, 'peq_filters') and self.peq_filters:
                # Get y-axis limits from left plot
                ylim1 = self.ax1.get_ylim()
                self.ax2.set_ylim(ylim1)
            else:
                y_min = np.min(magnitude_db) - 3
                y_max = np.max(magnitude_db) + 3
                self.ax2.set_ylim(y_min, y_max)
            
            self.ax2.set_xlim(20, fs/2)
            
            # Calculate and display error if target response exists
            if hasattr(self, 'peq_filters') and self.peq_filters:
                converter = PEQtoFIR(fs=fs, num_taps=len(fir_coeffs))
                
                # Use the unified method to get final target response
                target_freq, target_db = converter.get_final_target_response(
                    self.peq_filters,
                    use_file_preamp=self.apply_file_preamp.get(),
                    use_auto_preamp=self.apply_auto_preamp.get()
                )
                
                # Interpolate target to match FIR frequencies
                target_interp = np.interp(w, target_freq, target_db)  # type: ignore[arg-type]
                
                # Calculate error
                error = magnitude_db - target_interp
                max_error = np.max(np.abs(error))
                rms_error = np.sqrt(np.mean(error**2))
                
                # Add error text
                self.ax2.text(0.02, 0.98, 'Max Error: {:.2f} dB\nRMS Error: {:.2f} dB'.format(max_error, rms_error),
                            transform=self.ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            self.canvas.draw()
            self.status_label.config(text=f"Verified: {os.path.basename(file_path)} ({len(fir_coeffs)} taps, {fs} Hz)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to verify FIR file: {str(e)}")

def main():
    root = TkinterDnD.Tk()
    app = PEQtoFIRGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()