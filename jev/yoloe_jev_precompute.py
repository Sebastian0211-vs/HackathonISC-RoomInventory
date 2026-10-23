import sys
import json
import jev_classifier as jev

def compute_elem(element: str):
    return jev.classify(element)
    

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "./yoloe_names.txt"
    print("Reading file:", filename)

    products = []

    with open(filename, "r") as f:
        for line in f:
            element = line.strip()
            if element:
                result = compute_elem(element)
                products.append(result)
                print(f"{result}")

    # Save results to a JSON file
    output_filename = "classification_results.json"
    with open(output_filename, "w") as outfile:
        json.dump(products, outfile, indent=2)


        