import customtkinter as ctk
import qai_hub as hub
import threading

from PIL import Image
from tkinter import messagebox
from one_script_to_rule_them_all import inference_dataset, process_results, calculate_accuracy, InputSpec, extract_number, FileType
from os import listdir, path

class BenchmarkQAIHub():
	def __init__(self):
		self.root = ctk.CTk()
		self.root.geometry("500x500")
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

		self.root.mainloop()

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
				else:
					# Model ID
					model_id = self.model_id_entry.get()
					# Get Model
					model = hub.get_model(model_id)
				
				# Model Name
				model_name = path.splitext(model.name)[0]
				print(f"Model Name: {model_name}")

				# Library Name
				model_library = model.model_type.name.lower()
				print(f"Model Library: {model_library}")

				# Get Device Name
				device_name = self.device_name_entry.get()
				print(f"Device Name: f{device_name}")

				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"
				print(f"Results Directory: {results_dir}")

				### Run inference on image datasets
				# Dataset Paths
				datasets_dir = self.datasets_dir_entry.get()
				dataset_paths = [
					f"./{datasets_dir}/" + image_dataset 
					for image_dataset in listdir(f"./{datasets_dir}")
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

				thread1.start()
				thread2.start()

				thread1.join()
				thread2.join()

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
				messagebox.showinfo(title="❌ Error!", message="Bro. Btw. Check the textbox inputs if that other error message made zero sense. 👍")
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
				if self.model_path_entry.get():
					# Upload Model
					model_path = self.model_path_entry.get()
					model = hub.upload_model(model_path)	
				else:
					# Model ID
					model_id = self.model_id_entry_2.get()
					print(f"Model ID: {model_id}")
					# Get Model
					model = hub.get_model(str(model_id))
				
				# Model Name
				model_name = path.splitext(model.name)[0]
				print(f"Model Name: {model_name}")

				# Library Name
				model_library = model.model_type.name.lower()
				print(f"Model Library: {model_library}")

				# Get Device Name
				device_name = self.device_name_entry_2.get()
				print(f"Device Name: f{device_name}")

				# Set benchmark results directory
				model_device = device_name.replace(" ", "").lower()
				results_dir = f"{model_name}_{model_library}_{model_device}"
				print(f"Results Directory: {results_dir}")

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
				messagebox.showerror(title="❌ Error!", message="Bro. Btw. Check the textbox inputs if that other error message made zero sense. 👍")
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

BenchmarkQAIHub()