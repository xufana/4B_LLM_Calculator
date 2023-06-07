import json
import os.path as osp
import random
from typing import Union

def addition():
    # Addition up to 16 digits

    pairs = \
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(1,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(3,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(6,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(9,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(12,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(15,20) for j in range(i,20) for k in range(1000)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(18,20) for j in range(i,20) for k in range(1000)]

    random.shuffle(pairs)

    print("Addition:", len(pairs))

    data_add = []

    for num1, num2 in pairs:
    
        if random.random()<0.5:
            num1, num2 = num2, num1 

        answer = num1 + num2
    
        question = f"{num1} + {num2}" 
        output = f"{num1} + {num2} = {answer}"
    
        assert(output.split()[-1] == str(answer))
        data_add.append({"input": question, "output": output, "answer": str(answer)})
    
    
    
    return data_add

def subtraction():
    pairs = \
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(1,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(3,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(6,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(9,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(12,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(15,20) for j in range(i,20) for k in range(100)] +\
    [(random.randint(10**(i-1), 10**i), random.randint(10**(j-1), 10**j)) for i in range(18,20) for j in range(i,20) for k in range(100)]

    random.shuffle(pairs)

    print("Subtraction:", len(pairs))

    data_sub = []

    for num1, num2 in pairs:
    
        if random.random()<0.5:
            num1, num2 = num2, num1 

        answer = num1 - num2
    
        question = f"{num1} - {num2}" 
        output = f"{num1} - {num2} = {answer}"

        assert(output.split()[-1] == str(answer))
        data_sub.append({"input": question, "output": output, "answer": str(answer)})
    
    return data_sub

def main():
    data_add = addition()
    data_sub = subtraction()

    # dump arithmetic data into json

    template_name = "./templates/goat.json"
    data_path = "./data/"
    dataset_name = "dataset.json"

    with open(data_path + "addition.json", "w") as f:
        json.dump(data_add, f, indent=4)
    
    with open(data_path + "subtraction.json", "w") as f:
        json.dump(data_sub, f, indent=4)

    with open(data_path + dataset_name, "w") as f:
        json.dump(data_add + data_sub, f, indent=4)

    print("Total:", len(data_add + data_sub))
    
    print("Adding instructions and noise")

    ### Add natural language instruction to the generated arithmetic data using template

    with open(template_name) as fp:
        template = json.load(fp)

    with open(data_path + dataset_name, "rb") as test_file:
        data_original = json.load(test_file)

    data_converted = []

    for instance in data_original:
    
        arithmetic = instance["input"]
    
        output_dict = {}
        
        # add noise to instruction so that the model is robust to diverse question formats 
        if random.random() < 0.05:
            if " + " in arithmetic:
                arithmetic = "the sum of " + arithmetic.replace("+", "and")

            if " - " in arithmetic:
                arithmetic = "the difference of " + arithmetic.replace("-", "and")

        if random.random() < 0.5:
            arithmetic = arithmetic.replace("*", "x")    

        if random.random() < 0.1:
            arithmetic = arithmetic.replace("+", "plus").replace("-", "minus")   

        if random.random() < 0.5:
            if "+" in arithmetic or "-" in arithmetic:
                arithmetic = arithmetic.replace(" ", "")        

        num = random.randint(1,500)

        instruction = template[str(num)].format(
            input = arithmetic
        )
    
        output_dict["instruction"] = instruction
        output_dict["input"] = instance["input"]
        output_dict["output"] = instance["output"]
        output_dict["answer"] = instance["answer"]
    
        data_converted.append(output_dict)

    print("Total:", len(data_converted))

    with open(data_path + dataset_name, "w") as f:
        json.dump(data_converted, f, indent=4)

    print("Dataset generated!")

if __name__ == "__main__":
    main()
