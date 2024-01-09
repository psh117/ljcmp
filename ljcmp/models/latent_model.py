
class LatentValidityModel(object):
    def __init__(self) -> None:
        self.threshold = 0.5    

    def is_valid_estimated(self, z):
        raise NotImplementedError()

class LatentModel(object):
    def __init__(self, distribution) -> None:
        """_summary_

        Args:
            distribution (string): vmf or normal
        """
        self.distribution = None

    def to_latent(self, q):
        raise NotImplementedError()

    def to_state(self, z):
        raise NotImplementedError()

    def sample(self, num_samples):
        raise NotImplementedError()

    def sample_with_estimated_validity(self, num_samples:int, model:LatentValidityModel):
        raise NotImplementedError()
        
    def sample_with_estimated_validity_with_q(self, num_samples:int, model:LatentValidityModel):
        raise NotImplementedError()

    def predict(self, z):
        raise NotImplementedError()
    
class CoMPNetModelBase(object):
    def __init__(self) -> None:
        
        self.given_voxel = None
        self.Z_o = None
        self.Z_c = None

    def predict(self, cur_q, target_q):
        """
        returns next_q
        """
        raise NotImplementedError()
        
class MultiLatentModle(object):
    def __init__(self, z_dim = [0]) -> None:
        self.z_dim = z_dim

    def to_latent(self, q):
        raise NotImplementedError()

    def to_state(self, z):
        raise NotImplementedError()

    def sample(self, num_samples):
        raise NotImplementedError()

    def sample_with_estimated_validity(self, num_samples:int, depth_level:int, model:LatentValidityModel):
        raise NotImplementedError()
        
    def sample_with_estimated_validity_with_q(self, num_samples:int, depth_level:int, model:LatentValidityModel):
        raise NotImplementedError()

    def predict(self, z):
        raise NotImplementedError()
