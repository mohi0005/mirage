import os
import json
from collections import Counter

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
total = sum(data)
average = total / len(data)

counts = Counter(["a", "b", "a", "c", "b", "a"])
top = counts.most_common(3)

output = {"average": average, "top_items": top}
print(json.dumps(output, indent=2))
