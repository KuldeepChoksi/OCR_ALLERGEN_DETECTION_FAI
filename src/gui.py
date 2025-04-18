import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import os
import subprocess
import pandas as pd
import re
import ast  

class AllergenDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Allergen Detector")
        self.root.geometry("1420x900")  
        self.root.configure(bg="#f5f5f5")
        

        
        # Main frame
        self.main_frame = tk.Frame(root, bg="#f5f5f5")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(self.main_frame, text="Food Allergen Detector", font=("Arial", 38, "bold"), bg="#f5f5f5", fg="#333333")
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_label = tk.Label(self.main_frame, text="Upload a food label image to detect ingredients and potential allergens", 
                             font=("Arial", 20), bg="#f5f5f5", fg="#555555", wraplength=800)
        desc_label.pack(pady=(0, 20))
        
        # frame for the upload section
        self.upload_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        self.upload_frame.pack(fill=tk.X, pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(self.upload_frame, text="Upload Image", command=self.upload_image,
                                  font=("Arial", 16), bg="#000000", fg="black", padx=20, pady=10,
                                  relief=tk.RAISED, bd=3, activebackground="#333333", cursor="hand2")
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # File path label
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        self.file_path_label = tk.Label(self.upload_frame, textvariable=self.file_path_var, 
                                       font=("Arial", 14), bg="#f5f5f5", fg="#555555")
        self.file_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # horizontal divider
        ttk.Separator(self.main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Bottom section with image above results
        self.content_frame = tk.Frame(self.main_frame, bg="#f5f5f5")
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image frame 
        self.image_frame = tk.LabelFrame(self.content_frame, text="Image", font=("Arial", 14, "bold"), 
                                       bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        self.image_frame.pack(fill=tk.X, expand=False, pady=(0, 20), ipady=10)
        
        # Image placeholder 
        self.image_placeholder = tk.Label(self.image_frame, text="Image will appear here", 
                                         bg="#cccccc", fg="#555555", height=8, font=("Arial", 14))
        self.image_placeholder.pack(fill=tk.BOTH, expand=True)
        
        # Results frame 
        self.results_frame = tk.LabelFrame(self.content_frame, text="Results", font=("Arial", 14, "bold"), 
                                         bg="#f5f5f5", fg="#333333", padx=10, pady=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook 
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: OCR Results
        self.ocr_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.ocr_tab, text="OCR Results")
        
        self.ocr_text = scrolledtext.ScrolledText(self.ocr_tab, wrap=tk.WORD, 
                                               font=("Arial", 20), bg="white", fg="black", height=10)
        self.ocr_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 2: Allergens
        self.allergens_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.allergens_tab, text="Allergens")
        
        # Create treeview for allergens
        self.allergens_tree = ttk.Treeview(self.allergens_tab, columns=("Ingredient", "Class", "Type", "Group", "Allergy"), 
                                          show="headings", height=10)
        
        # Define headings
        self.allergens_tree.heading("Ingredient", text="Ingredient")
        self.allergens_tree.heading("Class", text="Class")
        self.allergens_tree.heading("Type", text="Type")
        self.allergens_tree.heading("Group", text="Group")
        self.allergens_tree.heading("Allergy", text="Allergy")
        
        # Define columns
        self.allergens_tree.column("Ingredient", width=100)
        self.allergens_tree.column("Class", width=100)
        self.allergens_tree.column("Type", width=120)
        self.allergens_tree.column("Group", width=120)
        self.allergens_tree.column("Allergy", width=150)
        
        # scrollbar
        scrollbar = ttk.Scrollbar(self.allergens_tab, orient=tk.VERTICAL, command=self.allergens_tree.yview)
        self.allergens_tree.configure(yscroll=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.allergens_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tab 3: Complex Ingredients
        self.complex_tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.complex_tab, text="Complex Ingredients")
        
        # Create treeview for complex ingredients
        self.complex_tree = ttk.Treeview(self.complex_tab, columns=("Ingredient", "Complex Name", "Simple Synonym"), 
                                        show="headings", height=10)
        
        # Define headings
        self.complex_tree.heading("Ingredient", text="Detected As")
        self.complex_tree.heading("Complex Name", text="Complex Name")
        self.complex_tree.heading("Simple Synonym", text="Simple Meaning")
        
        # Define columns
        self.complex_tree.column("Ingredient", width=120)
        self.complex_tree.column("Complex Name", width=150)
        self.complex_tree.column("Simple Synonym", width=200)
        
        # scrollbar
        complex_scrollbar = ttk.Scrollbar(self.complex_tab, orient=tk.VERTICAL, command=self.complex_tree.yview)
        self.complex_tree.configure(yscroll=complex_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.complex_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        complex_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                                 font=("Arial", 12), bg="#f0f0f0", fg="#333333", padx=10, pady=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.current_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self.status_var.set("Processing image...")
            self.root.update()
            
            self.display_image(file_path)
            
            # Process image
            try:
                self.process_image(file_path)
                self.status_var.set("Processing complete")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                self.status_var.set("Error in processing")

    def display_image(self, file_path):
        # Remove previous image if any
        for widget in self.image_placeholder.winfo_children():
            widget.destroy()
        
        # Open and resize image to fit the frame
        try:
            img = Image.open(file_path)
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            max_width = self.image_frame.winfo_width() - 40  # More horizontal space
            max_height = 300  # Fixed height for landscape orientation
            
            # Default dimensions if the frame is not yet properly sized
            if max_width < 50:
                max_width = 800  # Wider default
            
            # Calculate scaling factor
            width_ratio = max_width / width
            height_ratio = max_height / height
            ratio = min(width_ratio, height_ratio)
            
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Create label and display image
            image_label = tk.Label(self.image_placeholder, image=photo, bg="#cccccc")
            image_label.image = photo  # Keep a reference
            image_label.pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")

    def process_image(self, file_path):
        # Clear previous results
        self.ocr_text.delete(1.0, tk.END)
        for item in self.allergens_tree.get_children():
            self.allergens_tree.delete(item)
        for item in self.complex_tree.get_children():
            self.complex_tree.delete(item)
            
        # Set up text formatting tags
        self.ocr_text.tag_configure("heading", font=("Arial", 22, "bold"), foreground="black")
        self.ocr_text.tag_configure("normal", font=("Arial", 20), foreground="black")
        
        # Run the OCR script with the image path
        try:
            # Show processing status
            self.status_var.set("Running OCR analysis...")
            self.root.update()
            
            # Run the OCR script and capture output
            result = subprocess.run(['python', 'ocr.py', file_path], 
                                   capture_output=True, text=True)
            output = result.stdout
            error_output = result.stderr
            
            # For debugging - print output to console
            print("OCR Output:", output[:500] + "..." if len(output) > 500 else output)
            if error_output:
                print("Error Output:", error_output)
            
            # Parse the output
            self.status_var.set("Parsing results...")
            self.root.update()
            
            # Directly extract the OCR words for display in case parsing fails
            ocr_words_direct = []
            if "OCR Extracted Raw Words" in output:
                # First try the regex approach
                self.parse_output(output)
                
                # As a fallback, manually extract and display the words
                ocr_line = [line for line in output.split('\n') if "OCR Extracted Raw Words" in line]
                if ocr_line and not self.ocr_text.get("1.0", tk.END).strip():
                    try:
                        # Extract everything after -->
                        words_part = ocr_line[0].split('-->')[1].strip()
                        # Display raw output if parsing failed
                        self.ocr_text.delete(1.0, tk.END)
                        self.ocr_text.insert(tk.END, "Detected Words:\n\n", "heading")
                        self.ocr_text.insert(tk.END, words_part, "normal")
                    except Exception as e:
                        print(f"Fallback extraction failed: {e}")
                        self.ocr_text.delete(1.0, tk.END)
                        self.ocr_text.insert(tk.END, f"OCR Output:\n\n", "heading")
                        self.ocr_text.insert(tk.END, output[:1000], "normal")
            else:
                # If no structured output is found, display raw output
                self.ocr_text.delete(1.0, tk.END)
                self.ocr_text.insert(tk.END, f"OCR Raw Output:\n\n", "heading")
                self.ocr_text.insert(tk.END, output[:1000], "normal")
            
            self.status_var.set("Processing complete")
            
        except Exception as e:
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            # Display error in OCR text area for debugging
            self.ocr_text.insert(tk.END, f"Error occurred:\n", "heading")
            self.ocr_text.insert(tk.END, str(e), "normal")

    def parse_output(self, output):
        # Display raw OCR words
        ocr_words_match = re.search(r"OCR Extracted Raw Words -->\s+(\[.*?\])", output)
        if ocr_words_match:
            ocr_words_text = ocr_words_match.group(1)
            self.ocr_text.insert(tk.END, "Detected Words:\n\n", "heading")
            
            # Set tag configurations for headings and normal text
            self.ocr_text.tag_configure("heading", font=("Arial", 22, "bold"), foreground="black")
            self.ocr_text.tag_configure("normal", font=("Arial", 20), foreground="black")
            
            # Convert string representation of list to actual list
            try:
                ocr_words = ast.literal_eval(ocr_words_text)
                formatted_words = ", ".join(ocr_words)
                self.ocr_text.insert(tk.END, formatted_words, "normal")
            except Exception as e:
                # If parsing fails, just show the raw text
                self.ocr_text.insert(tk.END, ocr_words_text, "normal")
                print(f"Error parsing OCR words: {str(e)}")
        else:
            try:
                self.ocr_text.tag_configure("heading", font=("Arial", 22, "bold"), foreground="black")
                self.ocr_text.tag_configure("normal", font=("Arial", 20), foreground="black")
                
                ocr_line = [line for line in output.split('\n') if "OCR Extracted Raw Words" in line]
                if ocr_line:
                    words_part = ocr_line[0].split('-->')[1].strip()
                    self.ocr_text.insert(tk.END, "Detected Words:\n\n", "heading")
                    self.ocr_text.insert(tk.END, words_part, "normal")
                else:
                    # No matching line found
                    self.ocr_text.insert(tk.END, "Could not extract OCR words. Raw output:\n\n", "heading")
                    self.ocr_text.insert(tk.END, output[:1000], "normal")  # Show first 1000 chars
            except Exception as e:
                # If all else fails, show raw output
                self.ocr_text.insert(tk.END, f"Error parsing output: {str(e)}\n\nRaw output:\n", "heading")
                self.ocr_text.insert(tk.END, output[:1000], "normal")  # Show first 1000 chars
        
        # Parse allergens
        allergen_section = re.search(r"Allergens Warning:(.*?)(?:\n\n\nMapping Complex Ingredients|$)", 
                                    output, re.DOTALL)
        if allergen_section:
            allergen_text = allergen_section.group(1).strip()
            allergen_entries = re.findall(r"OCR word '(.+?)' matched with FoodData CSV entry '(.+?)'\.\s+"
                                        r"Class : (.+?)\s+Type\s+: (.+?)\s+Group: (.+?)\s+Allergy: (.+?)\s*\n",
                                        allergen_text)
            
            for i, entry in enumerate(allergen_entries):
                ocr_word, matched_food, food_class, food_type, food_group, allergy = entry
                self.allergens_tree.insert("", tk.END, values=(matched_food, food_class, food_type, food_group, allergy))
        
        # Parse complex ingredients 
        complex_section = re.search(r"Mapping Complex Ingredients to simpler synonyms:(.*?)$", 
                                   output, re.DOTALL)
        
        if complex_section:
            complex_text = complex_section.group(1).strip()
            # Print for debugging
            print("Complex section found:", complex_text[:100] + "..." if len(complex_text) > 100 else complex_text)
            
            # Improved regex pattern that handles end of string better
            complex_entries = re.findall(r"OCR word '(.+?)' matched with Complex Ingredient entry '(.+?)'\.\s+"
                                       r"Complex Ingredient : (.+?)\s+Simpler Synonym\s+: (.+?)(?:\s*\n\n|\s*$)",
                                       complex_text)
            
            print(f"Found {len(complex_entries)} complex ingredients")
            
            for i, entry in enumerate(complex_entries):
                ocr_word, complex_ingredient, complex_name, simple_synonym = entry
                print(f"Adding complex ingredient {i+1}: {ocr_word} -> {complex_name} -> {simple_synonym}")
                self.complex_tree.insert("", tk.END, values=(ocr_word, complex_name, simple_synonym))

    def run(self):
        self.upload_btn.bind("<Button-1>", lambda event: self.upload_image())
        
        # Set custom cursor for button to indicate clickability
        self.upload_btn.config(cursor="hand2")
        
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = AllergenDetectorGUI(root)
    app.run()