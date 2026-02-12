import csv

def make_recipes_txt(f_name = 'dump/13k-recipes.csv', output_file = "recipes.txt"):
    with open(output_file,'w',encoding='utf-8') as out_f:
        with open(f_name,'r',encoding='utf8') as f:
            r = csv.reader(f)
            next(r, None) #skip header
            i = 0
            for row in r:
                ingredients_cleaned = ", ".join(row[5][2:-2].split("', '"))
                outstr = f"Here is how to make {row[1]}. You need {ingredients_cleaned}. {row[3]}\n"
                out_f.writelines([outstr])
            

if __name__ == "__main__":
    make_recipes_txt()