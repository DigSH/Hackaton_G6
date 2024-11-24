#Librerías
library(tidyverse)

#Importe de datos
hotel <- read.csv2("data/hoteles.csv", sep = ",")
# Seleccionar variables
hotel <- hotel[, c("Municipio", 
                                "Nombre.Comercial", 
                                "Subcategoria", 
                                "Direccion.Comercial", 
                                "Correo.Electronico", 
                                "Habitaciones", 
                                "Camas", 
                                "Empleados")]

# Filtrar solo hoteles
hotel <- hotel[hotel$Subcategoria == "HOTEL", ]
hotel <- hotel[hotel$Camas > 0, ]
hotel <- hotel[hotel$Empleados > 0, ]

# Simulando la variable Precio
set.seed(123)  # Para reproducir los resultados
hotel$Precio <- round(50000 + (hotel$Empleados * 2000) + runif(nrow(hotel), min = -10000, max = 10000))

#
str(hotel)

# Exportar dataframe a CSV
write.csv(hotel, "hoteles.csv")


