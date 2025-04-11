import customtkinter as ctk
import qai_hub as hub
import threading
import pathlib
import time

from PIL import Image
from tkinter import messagebox
from one_script_to_rule_them_all import inference_dataset, process_results, calculate_accuracy, InputSpec, extract_number, FileType, inference_datasets_using_id
from os import listdir, path

class BenchmarkQAIHub():
	def __init__(self):
		self.root = ctk.CTk()
		self.root.geometry("500x600")
		self.root.title("Benchmark QAI Hub")
		self.root.iconbitmap("favicon.ico")
		
		# Title

		self.appraise_image = Image.open("Appraise.png")
		self.title_image = ctk.CTkImage(
			light_image=self.appraise_image,
			dark_image=self.appraise_image,
			size=(308,85)
		)
		
		self.title_label = ctk.CTkLabel(
			self.root, text="",
			image=self.title_image
		)
		self.title_label.pack()

		self.window_sub_label = ctk.CTkLabel(
			self.root,
			text="Benchmarking Tool",
			font=("Arial", 18))
		self.window_sub_label.pack(pady=(0, 20))

		# Create Tabs

		self.tabs = ctk.CTkTabview(self.root, width=500, height=500)
		self.tabs.pack()
		self.batch_inference_tab = self.tabs.add("Batch Datasets")
		self.single_inference_tab = self.tabs.add("Single Dataset")
		self.batch_inference_no_upload = self.tabs.add("Batch No Dataset Upload")

		### Batch Dataset Inference

		# Model ID
		self.model_id_var = ctk.StringVar()
		self.model_id_var.trace_add("write", self.check_entry)

		self.model_id_label = ctk.CTkLabel(
			self.batch_inference_tab,
			text="Model ID:",
			font=("Arial", 12)
		)
		self.model_id_label.pack()

		self.model_id_entry = ctk.CTkEntry(
			self.batch_inference_tab,
			font=("Arial", 16),
			width=300,
			textvariable=self.model_id_var
		)
		self.entry_original_color = self.model_id_entry._fg_color
		self.model_id_entry.pack()

		# Model Path
		self.model_path_var = ctk.StringVar()
		self.model_path_var.trace_add("write", self.check_entry)

		self.model_path_label = ctk.CTkLabel(
			self.batch_inference_tab,
			text="Model File Path:",
			font=("Arial", 12)
		)
		self.model_path_label.pack()

		self.model_path_entry = ctk.CTkEntry(
			self.batch_inference_tab,
			font=("Arial", 16),
			width=300,
			textvariable=self.model_path_var
		)
		self.model_path_entry.pack()

		# Device Name
		self.device_name_label = ctk.CTkLabel(
			self.batch_inference_tab,
			text="Device Name:", 
			font=("Arial", 12))
		self.device_name_label.pack()

		self.device_name_entry = ctk.CTkEntry(
			self.batch_inference_tab,
			font=("Arial", 16),
			width=300)
		self.device_name_entry.pack()

		# Datasets Directory
		self.datasets_dir_label = ctk.CTkLabel(
			self.batch_inference_tab,
			text="Datasets Directory Path:",
			font=("Arial", 12))
		self.datasets_dir_label.pack()

		self.datasets_dir_entry = ctk.CTkEntry(
			self.batch_inference_tab,
			font=("Arial", 16), width=300)
		self.datasets_dir_entry.pack()

		# Run Benchmark
		self.button_icon = Image.open("running-icon.png")

		self.run_benchmark_button = ctk.CTkButton(
			self.batch_inference_tab,
			text="Run Benchmark",
			command=self.run_batch_benchmark_threaded,
			image=ctk.CTkImage(
				dark_image=self.button_icon,
				light_image=self.button_icon
			)
		)
		self.run_benchmark_button.pack(pady=25)

		# Progress bar 1
		self.progressbar = ctk.CTkProgressBar(
			self.batch_inference_tab,
			orientation="horizontal",
			determinate_speed=2,
			width=300,
		)
		self.progressbar.set(0)
		self.progressbar.pack(pady=10)

		self.progressbar_label = ctk.CTkLabel(
			self.batch_inference_tab,
			text="0 / 50",
			font=("Arial", 12))
		self.progressbar_label.pack()

		### Single Dataset Inference

		# Model ID 2
		self.model_id_2_var = ctk.StringVar()
		self.model_id_2_var.trace_add("write", self.check_entry)

		self.model_id_label_2 = ctk.CTkLabel(
			self.single_inference_tab,
			text="Model ID:",
			font=("Arial", 12)
		)
		self.model_id_label_2.pack()

		self.model_id_entry_2 = ctk.CTkEntry(
			self.single_inference_tab,
			font=("Arial", 16), width=300,
			textvariable=self.model_id_2_var	
		)
		self.model_id_entry_2.pack()

		# Model Path 2
		self.model_path_2_var = ctk.StringVar()
		self.model_path_2_var.trace_add("write", self.check_entry)

		self.model_path_label_2 = ctk.CTkLabel(
			self.single_inference_tab,
			text="Model File Path:",
			font=("Arial", 12)
		)
		self.model_path_label_2.pack()

		self.model_path_entry_2 = ctk.CTkEntry(
			self.single_inference_tab,
			font=("Arial", 16),
			width=300,
			textvariable=self.model_path_2_var	
		)
		self.model_path_entry_2.pack()

		# Device Name 2
		self.device_name_label_2 = ctk.CTkLabel(
			self.single_inference_tab,
			text="Device Name:", 
			font=("Arial", 12))
		self.device_name_label_2.pack()

		self.device_name_entry_2 = ctk.CTkEntry(
			self.single_inference_tab,
			font=("Arial", 16),
			width=300)
		self.device_name_entry_2.pack()

		# Dataset Path
		self.dataset_path_label = ctk.CTkLabel(
			self.single_inference_tab,
			text="Dataset Path:",
			font=("Arial", 12))
		self.dataset_path_label.pack()

		self.dataset_path_entry = ctk.CTkEntry(
			self.single_inference_tab,
			font=("Arial", 16), width=300)
		self.dataset_path_entry.pack()

		# Run Benchmark 2
		self.button_icon = Image.open("running-icon.png")

		self.run_inference_button = ctk.CTkButton(
			self.single_inference_tab,
			text="Run Inference",
			command=self.run_dataset_inference_threaded,
			image=ctk.CTkImage(
				dark_image=self.button_icon,
				light_image=self.button_icon
			)
		)
		self.button_color = self.run_inference_button._fg_color
		self.run_inference_button.pack(pady=25)

		### Batch No Dataset Upload

		# Model ID
		self.model_id_3_var = ctk.StringVar()
		self.model_id_3_var.trace_add("write", self.check_entry)

		self.model_id_label_3 = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="Model ID:",
			font=("Arial", 12)
		)
		self.model_id_label_3.pack()

		self.model_id_entry_3 = ctk.CTkEntry(
			self.batch_inference_no_upload,
			font=("Arial", 16),
			width=300,
			textvariable=self.model_id_3_var
		)
		self.model_id_entry_3.pack()

		# Model Path
		self.model_path_3_var = ctk.StringVar()
		self.model_path_3_var.trace_add("write", self.check_entry)

		self.model_path_label_3 = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="Model File Path:",
			font=("Arial", 12)
		)
		self.model_path_label_3.pack()

		self.model_path_entry_3 = ctk.CTkEntry(
			self.batch_inference_no_upload,
			font=("Arial", 16),
			width=300,
			textvariable=self.model_path_3_var
		)
		self.model_path_entry_3.pack()

		# Device Name
		self.device_name_label_3 = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="Device Name:", 
			font=("Arial", 12))
		self.device_name_label_3.pack()

		self.device_name_entry_3 = ctk.CTkEntry(
			self.batch_inference_no_upload,
			font=("Arial", 16),
			width=300)
		self.device_name_entry_3.pack()

		# File Type
		self.optionmenu_label = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="File Type:", 
			font=("Arial", 12))
		self.optionmenu_label.pack()

		self.optionmenu_var = ctk.StringVar()
		self.optionmenu = ctk.CTkOptionMenu(
			self.batch_inference_no_upload,
			values=["tflite", "onnx"],
			variable=self.optionmenu_var
		)
		self.optionmenu.pack()

		# Input Spec
		self.optionmenu_label_2 = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="Input Spec:", 
			font=("Arial", 12))
		self.optionmenu_label_2.pack()

		self.optionmenu_2_var = ctk.StringVar()
		self.optionmenu_2 = ctk.CTkOptionMenu(
			self.batch_inference_no_upload,
			values=["normal", "quantized"],
			variable=self.optionmenu_2_var
		)
		self.optionmenu_2.pack()

		# Run Benchmark 3
		self.button_icon = Image.open("running-icon.png")

		self.run_batch_no_dataset_upload_button = ctk.CTkButton(
			self.batch_inference_no_upload,
			text="Run Benchmark",
			command=self.run_batch_no_dataset_upload_benchmark_threaded,
			image=ctk.CTkImage(
				dark_image=self.button_icon,
				light_image=self.button_icon
			)
		)

		self.run_batch_no_dataset_upload_button.pack(pady=15)

		# Progress bar 3
		self.progressbar_3 = ctk.CTkProgressBar(
			self.batch_inference_no_upload,
			orientation="horizontal",
			determinate_speed=2,
			width=300,
		)
		self.progressbar_3.set(0)
		self.progressbar_3.pack(pady=10)

		self.progressbar_label_3 = ctk.CTkLabel(
			self.batch_inference_no_upload,
			text="0 / 50",
			font=("Arial", 12))
		self.progressbar_label_3.pack()

		self.root.mainloop()

	def update_progressbar(self, progressbar, label, results_dir, total=50):
		while True:
			current_files = listdir(results_dir)
			progress = min(len(current_files) / total, 1.0)

			self.root.after(0, progressbar.set, progress)
			self.root.after(0, lambda: label.configure(text=f"{len(current_files)} / {total}"))

			if progress >= 1.0:
				break

			time.sleep(0.5)

	def check_entry(self, *args):
		if self.model_id_var.get():
			self.model_path_entry.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_path_entry.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

		if self.model_path_var.get():
			self.model_id_entry.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_id_entry.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

		if self.model_id_2_var.get():
			self.model_path_entry_2.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_path_entry_2.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

		if self.model_path_2_var.get():
			self.model_id_entry_2.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_id_entry_2.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

		if self.model_id_3_var.get():
			self.model_path_entry_3.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_path_entry_3.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

		if self.model_path_3_var.get():
			self.model_id_entry_3.configure(
				state="disabled",
				fg_color="dark grey"
			)
		else:
			self.model_id_entry_3.configure(
				state="normal",
				fg_color=self.entry_original_color
			)

	def run_batch_benchmark_threaded(self):
		# Run the benchmark in a separate thread to prevent UI freezing.
		threading.Thread(target=self.run_batch_benchmark, daemon=True).start()

	def run_batch_benchmark(self):
		if self.get_dataset_dir() and self.get_model_id_path() and self.get_device_name():
			try:
				self.run_benchmark_button.configure(state="disabled", fg_color="dark grey")  # Disable button
				if self.model_path_entry.get():
					# Upload Model
					model_path = self.model_path_entry.get()
					model = hub.upload_model(str(model_path))

					# Model ID
					model_id = model.model_id
				else:
					# Model ID
					model_id = self.model_id_entry.get()
					# Get Model
					model = hub.get_model(model_id)
				
				# Model Name
				model_name = path.splitext(model.name)[0]
				
				# Library Name
				model_library = model.model_type.name.lower()
				
				# Get Device Name
				device_name = self.device_name_entry.get()
				
				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"

				# Ensure folder exists
				results_dir = pathlib.Path(results_dir)
				results_dir.mkdir(parents=True, exist_ok=True)
				
				print(f"------------------------------------------------------")
				print(f"| Model ID: {model_id}")
				print(f"| Model Name: {model_name}")
				print(f"| Model Library: {model_library}")
				print(f"| Device Name: {device_name}")
				print(f"| Results Directory: {results_dir}")
				print(f"------------------------------------------------------")
				
				### Run inference on image datasets
				# Dataset Paths
				datasets_dir = self.datasets_dir_entry.get()
				dataset_paths = [
					f"{datasets_dir}/" + image_dataset 
					for image_dataset in listdir(f"{datasets_dir}")
				]
				dataset_paths.sort(key=extract_number)

				# Split into two lists with alternating elements
				list1 = dataset_paths[::2]  # odd
				list2 = dataset_paths[1::2]  # even

				thread1 = threading.Thread(
					target=inference_dataset,
					args=(list1, model_id, device_name, model_name, results_dir)
				)

				thread2 = threading.Thread(
					target=inference_dataset,
					args=(list2, model_id, device_name, model_name, results_dir)
				)

				thread3 = threading.Thread(
    				target=self.update_progressbar,
    				args=(self.progressbar, self.progressbar_label, results_dir)
				)

				thread1.start()
				thread2.start()
				thread3.start()

				thread1.join()
				thread2.join()
				thread3.join()

				### Process results from inference
				result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
				result_paths.sort(key=extract_number)
				process_results(result_paths, "./class_index.json", "./synset.json")

				### Calculate accuracy based on processed results
				results_json = "results.json"
				ground_truth_json = "ground_truth.json"

				calculate_accuracy(results_json, ground_truth_json, device_name, model_name, model_library)

				# Show success message after completion
				messagebox.showinfo(title="✅ Success!", message="Benchmark completed successfully!")

			except Exception as e:
				messagebox.showerror(title="❌ Error!", message=f"An error occurred: {e}")
			finally:
				self.run_benchmark_button.configure(state="normal", fg_color=self.button_color)  # Re-enable button
		else:
			print("😱 This should never happen. 😱")

	def get_model_id_path(self):
		model_id = self.model_id_entry.get()
		model_path = self.model_path_entry.get()
		if not model_id and not model_path:
			messagebox.showerror(title="❌ Error!", message="No model file path or model id given.")
			return False
		return True
	
	def get_device_name(self):
		device_name = self.device_name_entry.get()
		if not device_name:
			messagebox.showerror(title="❌ Error!", message="No device name given.")
			return False
		return True

	def get_dataset_dir(self):
		datasets_dir = self.datasets_dir_entry.get()
		if not datasets_dir:
			messagebox.showerror(title="❌ Error!", message="No datasets directory path given.")
			return False
		return True

	def run_dataset_inference_threaded(self):
		# Run the benchmark in a separate thread to prevent UI freezing.
		threading.Thread(target=self.run_dataset_inference, daemon=True).start()

	def run_dataset_inference(self):
		if self.get_dataset_path() and self.get_model_id_path_2() and self.get_device_name_2():
			try:
				self.run_inference_button.configure(state="disabled", fg_color="dark grey")  # Disable button
				if self.model_path_entry_2.get():
					# Upload Model
					model_path = self.model_path_entry_2.get()
					model = hub.upload_model(model_path)

					# Model ID
					model_id = model.model_id
				else:
					# Model ID
					model_id = self.model_id_entry_2.get()
					# Get Model
					model = hub.get_model(str(model_id))
				
				# Model Name
				model_name = path.splitext(model.name)[0]

				# Library Name
				model_library = model.model_type.name.lower()

				# Get Device Name
				device_name = self.device_name_entry_2.get()

				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"

				print(f"------------------------------------------------------")
				print(f"| Model ID: {model_id}")
				print(f"| Model Name: {model_name}")
				print(f"| Model Library: {model_library}")
				print(f"| Device Name: {device_name}")
				print(f"| Results Directory: {results_dir}")
				print(f"------------------------------------------------------")

				### Run inference on image datasets
				dataset_paths = [self.dataset_path_entry.get()]
				inference_dataset(dataset_paths, model_id, device_name, model_name, results_dir)

				### Process results from inference
				result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
				result_paths.sort(key=extract_number)
				process_results(result_paths, "./class_index.json", "./synset.json")

				### Calculate accuracy based on processed results
				results_json = "results.json"
				ground_truth_json = "ground_truth.json"

				calculate_accuracy(results_json, ground_truth_json, device_name, model_name, model_library)

				# Show success message after completion
				messagebox.showinfo(title="✅ Success!", message="Benchmark completed successfully!")

			except Exception as e:
				messagebox.showerror(title="❌ Error!", message=f"An error occurred: {e}")
			finally:
				self.run_inference_button.configure(state="normal", fg_color=self.button_color)  # Re-enable button
		else:
			print("Check message box. 👍")

	def get_model_id_path_2(self):
		model_id = self.model_id_entry_2.get()
		model_path = self.model_path_entry_2.get()
		if not model_id and not model_path:
			messagebox.showerror(title="❌ Error!", message="No model file path or model id given.")
			return False
		return True
	
	def get_device_name_2(self):
		device_name = self.device_name_entry_2.get()
		if not device_name:
			messagebox.showerror(title="❌ Error!", message="No device name given.")
			return False
		return True

	def get_dataset_path(self):
		datasets_path = self.dataset_path_entry.get()
		if not datasets_path:
			messagebox.showerror(title="❌ Error!", message="No datasets directory path given.")
			return False
		return True

	def run_batch_no_dataset_upload_benchmark_threaded(self):
		# Run the benchmark in a separate thread to prevent UI freezing.
		threading.Thread(target=self.run_batch_no_dataset_upload_benchmark, daemon=True).start()

	def run_batch_no_dataset_upload_benchmark(self):
		if self.get_file_type_input_spec() and self.get_model_id_path_3() and self.get_device_name_3():
			try:
				self.run_batch_no_dataset_upload_button.configure(state="disabled", fg_color="dark grey")  # Disable button
				if self.model_path_entry_3.get():
					# Upload Model
					model_path = self.model_path_entry_3.get()
					model = hub.upload_model(str(model_path))

					# Model ID
					model_id = model.model_id
				else:
					# Model ID
					model_id = self.model_id_entry_3.get()
					# Get Model
					model = hub.get_model(model_id)
				
				# Model Name
				model_name = path.splitext(model.name)[0]
				
				# Library Name
				model_library = model.model_type.name.lower()
				
				# Get Device Name
				device_name = self.device_name_entry_3.get()
				
				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"

				# Ensure folder exists
				results_dir = pathlib.Path(results_dir)
				results_dir.mkdir(parents=True, exist_ok=True)
				
				print(f"------------------------------------------------------")
				print(f"| Model ID: {model_id}")
				print(f"| Model Name: {model_name}")
				print(f"| Model Library: {model_library}")
				print(f"| Device Name: {device_name}")
				print(f"| Results Directory: {results_dir}")
				print(f"------------------------------------------------------")
				
				### Run inference on image datasets
				tflite_normal = [
					'dv74k8ew2', 'dv910vo82', 'dq9krm657', 'd82ndxp57', 'dv9518om2',
					'dd9ppq5n9', 'dz7z43qr9', 'd67jwmon2', 'd67oxpon7', 'dp7lgeow2',
					'dk7gkz4o2', 'dv74k8qw2', 'dq9krmo57', 'dp70nz3l9', 'd82ndxo57',
					'd09y13p39', 'dv95185m2', 'dd9ppqon9', 'dn7xzrx59', 'dj7d0em89',
					'dz7z43vr9', 'dx9e8e0p9', 'd67jwm4n2', 'dw9v84vj7', 'd693mv6l7',
					'dp7lge3w2', 'dk7gkzmo2', 'dz2r543o7', 'dr9wme332', 'dv74k83w2',
					'dq9krm357', 'd678rge62', 'd09y13v39', 'dv95186m2', 'dr2qq53l2',
					'dj7d0en89', 'dr2qq51o2', 'dn7xzrnv9', 'dz7z43869', 'dx9e8e149',
					'dw9v84507', 'd67oxpmq7', 'd693mvwp7', 'dw264zde9', 'dk7gkzx02',
					'dr9wmeyk2', 'dv910vwe2', 'd678rgpy2', 'd09y130m9', 'dd9ppqew9',
				]

				tflite_quantized = []

				onnx_normal = [
					'dj7d0qwq9', 'dz7z4jny9', 'dx9e85mv9', 'dw9v8m6z7', 'd67oxrvl7',
					'd693mj007', 'dw264vqg9', 'dp7lgrv12', 'dk7gkn1e2', 'dz2r5veg7',
					'dr9wm6wo2', 'dv74krjz2', 'dv9103yx2', 'dq9krg1d7', 'dp70nmko9',
					'd09y1jmv9', 'dd9pprgm9', 'dj7d0q4q9', 'dz7z4j0y9', 'dw9v8mpz7',
					'd67oxrzl7', 'dz2r5v8g7', 'dv74kr0z2', 'do7ml6zl9', 'd82ndr4o7',
					'dr2qqmgv2', 'dn7xzly69', 'dz7z4joy9', 'd693mjn07', 'dw264vkg9',
					'dz2r56jg7', 'dr9wmp8o2', 'do7mle1l9', 'd678r4vm2', 'd82nd6wo7',
					'dd9ppz6m9', 'dn7xzqk69', 'dj7d03lp9', 'dz7z4yrw9', 'dw9v8xnq7',
					'dw264wj69', 'dz2r56g07', 'dv74kloy2', 'do7mlek39', 'dp70nlg09',
					'dv951jvz2', 'dr2qq6r62', 'dd9ppzwd9', 'dn7xzqjr9', 'dj7d03xp9'
				]

				onnx_quantized = []

				file_type = self.optionmenu.get()
				input_spec = self.optionmenu_2.get()

				if file_type == "tflite" and input_spec == "normal":
					dataset_ids = tflite_normal[::-1]
				elif file_type == "tflite" and input_spec == "quantized":
					dataset_ids = tflite_quantized[::-1]
				elif file_type == "onnx" and input_spec == "normal":
					dataset_ids = onnx_normal[::-1]
				elif file_type == "onnx" and input_spec == "quantized":
					dataset_ids = onnx_quantized[::-1]

				# Split into two lists with alternating elements
				list1 = dataset_ids[::2]  # odd
				list2 = dataset_ids[1::2]  # even

				thread1 = threading.Thread(
					target=inference_datasets_using_id,
					args=(list1, model_id, device_name, model_name, results_dir)
				)

				thread2 = threading.Thread(
					target=inference_datasets_using_id,
					args=(list2, model_id, device_name, model_name, results_dir)
				)

				thread3 = threading.Thread(
    				target=self.update_progressbar,
    				args=(self.progressbar_3, self.progressbar_label_3, results_dir)
				)
	
				thread1.start()
				thread2.start()
				thread3.start()

				thread1.join()
				thread2.join()
				thread3.join()

				### Process results from inference
				result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
				result_paths.sort(key=extract_number)
				process_results(result_paths, "./class_index.json", "./synset.json")

				### Calculate accuracy based on processed results
				results_json = "results.json"
				ground_truth_json = "ground_truth.json"

				calculate_accuracy(results_json, ground_truth_json, device_name, model_name, model_library)

				# Show success message after completion
				messagebox.showinfo(title="✅ Success!", message="Benchmark completed successfully!")

			except Exception as e:
				messagebox.showerror(title="❌ Error!", message=f"An error occurred: {e}")
			finally:
				self.run_batch_no_dataset_upload_button.configure(state="normal", fg_color=self.button_color)  # Re-enable button
		else:
			print("😱 This should never happen. 😱")

	def get_file_type_input_spec(self):
		file_type = self.optionmenu.get()
		input_spec = self.optionmenu_2.get()
		if not file_type or not input_spec:
			messagebox.showerror(title="❌ Error!", message="No file type or input spec given.")
			return False
		return True

	def get_model_id_path_3(self):
		model_id = self.model_id_entry_3.get()
		model_path = self.model_path_entry_3.get()
		if not model_id and not model_path:
			messagebox.showerror(title="❌ Error!", message="No model file path or model id given.")
			return False
		return True
	
	def get_device_name_3(self):
		device_name = self.device_name_entry_3.get()
		if not device_name:
			messagebox.showerror(title="❌ Error!", message="No device name given.")
			return False
		return True

BenchmarkQAIHub()