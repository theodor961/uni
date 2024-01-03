from enum import Enum
import tkinter as tk
import random 
import csv

FONT_SMALL = ("Segoe UI", "12", 'bold')
FONT_LARGE = ("Segoe UI", "18", 'bold')

class Grid:
    def __init__(self, dimensions, gridFromFile=None, maxReward = 10):
        self.dimensions = dimensions
        self.maxReward = maxReward
        
        self.rows = self.dimensions[0]
        self.cols = self.dimensions[1]
        
        if gridFromFile:
            self.readGridData(gridFromFile)
        else:
           self.generateGridData()
        
    def generateGridData(self):
        self.gridRewards = [random.randrange(-int(self.maxReward/2), self.maxReward, 1) for _ in range(self.cols * self.rows)]

    def saveGridData(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([self.rows, self.cols])
            csvwriter.writerow(self.gridRewards)
            
    def readGridData(self, filename):
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.rows, self.cols = map(int, next(csvreader))
            self.gridRewards = [int(value) for value in next(csvreader)]
        
    def drawGrid(self, cell_size=100):
        root = tk.Tk()
        root.title("Grid World")

        canvas = tk.Canvas(root, width=(self.cols+1) * cell_size, height=(self.rows+1) * cell_size)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(root, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.config(yscrollcommand=scrollbar.set)
        
        for i in range(1, self.rows+1):
            canvas.create_line(0, i * cell_size, (self.cols+1) * cell_size, i * cell_size, fill='black', width=2)
        for j in range(1, self.cols+1):
            canvas.create_line(j * cell_size, 0, j * cell_size, (self.rows+1) * cell_size, fill='black', width=2)

        canvas.create_text(0.5 * cell_size, 0.5 * cell_size, text="state (x,y)", font=FONT_SMALL, fill="red")
        
        for i in range (1, self.rows + 1):
            canvas.create_text(0.5 * cell_size, (i + 0.5) * cell_size, text=i, font=FONT_LARGE, fill='red')
        for i in range (1, self.cols +1):
            canvas.create_text( (i + 0.5) * cell_size, 0.5 * cell_size, text=i, font=FONT_LARGE, fill='red')
            
        rewardIndex = 0
        for i in range(1,self.rows +1):
            for j in range(1, self.cols+1):
                text = self.gridRewards[rewardIndex]
                rewardIndex+=1
                
                if rewardIndex != self.rows * self.cols and rewardIndex != 1:
                    canvas.create_text((j + 0.5) * cell_size, (i + 0.5) * cell_size, text=text, font=FONT_LARGE, fill='black')
                elif rewardIndex == self.rows * self.cols:
                    canvas.create_text((j + 0.5) * cell_size, (i + 0.5) * cell_size,text=f'Goal: {text}', font=FONT_LARGE, fill='red')
                else:
                    canvas.create_text((j + 0.5) * cell_size, (i + 0.5) * cell_size, text=f'Start: {text}', font=FONT_LARGE, fill='red')
                

        canvas.config(scrollregion=canvas.bbox("all"))
        root.mainloop()
        

Dimensions = (5,5)
grid = Grid(dimensions=Dimensions, gridFromFile=None, maxReward=10)
grid.saveGridData("grid.csv")

        
