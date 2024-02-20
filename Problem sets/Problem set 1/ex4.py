from PyPDF2 import PdfReader
import re
import os

pdf_path = "Advanced Appstat/Problem sets/Problem set 1/authors-acknowledgements-v5.pdf"

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages[0:11]:
    text += page.extract_text()
# Remove the lines that don't have any names
text = ''.join(text.splitlines(keepends=True)[4:-21])
# Remove MMA—LIGO... string at top of most pages
text = re.sub("(MMA\s*—\s*LIGO\s*-\s*P1700294\s*-\s*V5)", "", text)
# Remove all parentheses and the text in them (these are never names)
text = re.sub(r"\((.*?)\)", "", text)
# Replace all numbers with commas
text = re.sub(r"[0-9]+", ",", text)
# Remove AND (when it is used as the word) and replace with commas
text = re.sub(r"(\s*,\s*AND\s+)", ",", text)
# Replace all newlines with commas
text = text.replace("\n", ",")
# Remove all whitespaces
text = re.sub(r"\s+", "", text)
# Replace 2 or more commas in a row with one comma
text = re.sub(r"[,]{2,}",",", text)
# Remove all accents; aren't handled well after reading PDF.
text = re.sub(r"(`|´|~|¨|¸|˜|ˇ|˙)+", "", text)
# Insert space between name and "JR"
text = re.sub(r"(JR.)", " JR", text)
# Replace all . with . followed by a whitespace
text = re.sub(r"\.", ". ", text)
# Create name list. Last element is empty, so exclude it
name_list = text.split(",")[:-1]
# Get number of unique authors
n_unique = len(name_list)
# Print number of unique authors
print(f"Number of unique authors {n_unique}")
# Get list of last names
lname_list = [name.split(" ")[-1] for name in name_list]
# Get list of initials
initials_list = [" ".join(name.split(" ")[:-1]) for name in name_list]
# Create list of names starting with last name followed by initials
name_list_flipped = [lname + " " + initials for lname, initials in zip(lname_list, initials_list)]
# Sort the above list alphabetically
sorted_names_flipped = sorted(name_list_flipped)
# Get the author at the midway point in the sorted list
author_mid = sorted_names_flipped[int(n_unique/2)-1]
print(f"Author at midway point: {author_mid}")
# Write the sorted list to a file
with open(f"{os.getcwd()}/sorted_names.txt", "w") as f:
    f.write("\n".join(sorted_names_flipped))
