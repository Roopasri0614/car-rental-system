import chardet

with open(r"C:\Users\roopa\Downloads\movie\IMDb Movies India.csv", "rb") as f:
    result = chardet.detect(f.read(100000))
print("Detected encoding:", result['encoding'])
