import rpy2.robjects as robjects
import matplotlib.pyplot as plt
# Install the ggplot2 package if not already installed
#r('if (!requireNamespace("ggplot2", quietly = TRUE)) { install.packages("ggplot2") }')
import rpy2.robjects as robjects

# Load the Titanic dataset in R and convert it into an R dataframe
r = robjects.r
r('titanic_data <- read.csv("C:/Users/king/Desktop/jupyterythonassignment/Python_Assignment10/titanic/titanic.csv")')

# Print the head of the dataset
result_head = r('head(titanic_data)')
print(result_head)
result_structure = r('str(titanic_data)')
print(result_structure)




# Plot the histogram for Age distribution
r('hist(titanic_data$Age, main="Age Distribution", xlab="Age", col="blue")')
# Plot the bar plot for Passenger Class (Pclass) distribution
r('barplot(table(titanic_data$Pclass), main="Passenger Class Distribution", xlab="Passenger Class", ylab="Count", col="orange")')
# Plot the bar plot for Survival Status (Survived) distribution
r('barplot(table(titanic_data$Survived), main="Survival Status Distribution", xlab="Survived", ylab="Count", col="green")')



# Load and display the saved histogram plot
img = plt.imread("histogram.png")
plt.imshow(img)
plt.axis("off")
plt.show()





