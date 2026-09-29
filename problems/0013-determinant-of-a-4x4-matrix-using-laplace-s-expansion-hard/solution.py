def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	n=len(matrix)
	
	if n==1:
		return matrix[0][0]
	if n==2:
		return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
	det=0
	for col in range(n):
		minor=[
		row[:col]+row[col+1:] for row in matrix[1:]
		]
		sign=1 if col%2==0 else -1
		det+=sign*matrix[0][col]*determinant_4x4(minor)
	return det



