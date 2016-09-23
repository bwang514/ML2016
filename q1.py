import sys

def main():	
	file = open("hw0_data.dat")
	lines = file.readlines(500000)
	col = int(sys.argv[1]) + 1
	output = []
	if col == 11 :
		for line in lines:
			output.append(int(line.split(" ")[11].split("\n")[0]))
	else :		
		for line in lines:
			output.append(float(line.split(" ")[col]))
	
	output.sort()
	out = ','.join(map(str, output)) 
	f = open('ans1.txt','w')
	f.write(out) 

	

if __name__ == '__main__':
	main()




