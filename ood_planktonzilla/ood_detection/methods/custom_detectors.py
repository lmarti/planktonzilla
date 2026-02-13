from pytorch_ood.detector import ViM, KLMatching, Mahalanobis,EnergyBased, MaxLogit, ReAct, MaxSoftmax
import torch.nn as nn
import datetime 
import torch
#TODO: Para cada clase guardar en el mismo directorio la descripcion del experimento (config, modelo, etc) junto con los parametros ajustados para cada metodo. Esto facilita la trazabilidad y reproducibilidad de los experimentos.
class CustomEnergyBased(EnergyBased):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomMaxLogit(MaxLogit):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomReAct(ReAct):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomMaxSoftmax(MaxSoftmax):
    def save_fitted_parameters(self,save_path:str):
        return self
    def load_fitted_parameters(self,load_path:str):
        return self

class CustomKLMatching(KLMatching):
    def save_fitted_parameters(self,save_path:str):
        #Save fitted parameters for later use in evaluation 
        assert len(self.dists) !=0, "No fitted parameters to save. Please fit the detector first."
        torch.save(self.dists.state_dict(), f"{save_path}/kl_matching_parameters.pt")
    
    def load_fitted_parameters(self,load_path:str): 
        state_dict = torch.load(f"{load_path}/kl_matching_parameters.pt",map_location="cpu") 
        self.dists = nn.ParameterDict()
        for key, tensor in state_dict.items():
            self.dists[key] = nn.Parameter(tensor)

class CustomMahalanobis(Mahalanobis):
    def save_fitted_parameters(self,save_path:str): #Save fitted parameters
        #Save fitted parameters for later use in evaluation
         if self.mu is not None and self.precision is not None:
            d = {"mu":self.mu,
                "precision":self.precision,
                }
            torch.save(d, f"{save_path}/mahalanobis_parameters.pt") 
    def load_fitted_parameters(self,load_path:str): 
        params = torch.load(f"{load_path}/mahalanobis_parameters.pt") 
        self.mu = params["mu"] 
        self.precision = params["precision"]

class CustomViM(ViM):
    def save_fitted_parameters(self,save_path:str):
        #Save fitted parameters for later use in evaluation
        if self.alpha is not None and self.principal_subspace is not None:

            d = {
            "alpha":self.alpha,
            "principal_subspace":self.principal_subspace,
            }

            torch.save(d, f"{save_path}/vim_parameters.pt")

    def load_fitted_parameters(self,load_path:str): 
        params = torch.load(f"{load_path}/vim_parameters.pt",map_location="cpu", weights_only=False)
        self.alpha = params["alpha"] 
        self.principal_subspace = params["principal_subspace"]

