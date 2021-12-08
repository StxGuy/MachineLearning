TARGET = deepQ
F90_SOURCE =  ./source/activ.f90 \
              ./source/loss.f90 \
              ./source/layer.f90 \
              ./source/neural.f90 \
              ./source/deepQ.f90 \
              
	     
#$(wildcard ./source/*.f90)
OBJ = $(subst .f90,.o,$(subst source,objects,$(F90_SOURCE)))
FC = gfortran
FC_FLAGS = -O3 \
           -ffast-math \
           -llapack \
           -lblas \

MOD = mods           
           
RM = rm -rf           

all : display $(TARGET)
	

display:
	@clear
	@mkdir -p objects
	@mkdir -p mods
	
	@echo ".------------------------------------."
	@echo "| Compiling: DeepQ                   |#"
	@echo "| ---------                          |#"
	@echo "|                                    |#"
	@echo "| By: Prof. Carlo R. da Cunha        |#"
	@echo "|                                    |#"
	@echo "| Created: Nov/2021                  |#"
	@echo "| Revision: Dec/2021                 |#"
	@echo "'------------------------------------'#"
	@echo "  #####################################"
	@echo ""


$(TARGET): $(OBJ)
	@echo "# Linking $@..."
	$(FC) $^ $(FC_FLAGS) -o $@ -I $(MOD)
	@echo ""

./objects/%.o: ./source/%.f90
	@echo "# Building target: $<"
	$(FC) $< $(FC_FLAGS) -c -o $@ -J $(MOD) 
	@echo ""
		
	
clean:
	@$(RM) ./objects/*.o $(TARGET) *~
	@$(RM) ./mods/*.mod $(TARGET) *~
	@rmdir objects
	@rmdir mods
	
.PHONY: all clean	
	
	
