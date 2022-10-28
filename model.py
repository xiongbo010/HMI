import torch
import torch.nn as nn
import geoopt as gt
import itertools
import torch.nn
import torch.nn.functional
import geoopt.manifolds.stereographic.math as pmath
import torch.nn.functional as F

MIN_NORM = 1e-15
dropout = 0.5

cuda_device = torch.device('cuda:1')
    
class MobiusLinear(nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    k=-1.0,
):
    if hyperbolic_input:
        weight = F.dropout(weight, dropout)
        output = pmath.mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=k)
        output = pmath.mobius_add(output, bias, k=k)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = pmath.project(output, k=k)
    return output


from util.hyperop import poinc_dist
from util import hyperop
import geoopt.manifolds.stereographic.math as pmath

class HMI(nn.Module):
    
    def __init__(self, feature_num, hidden_size, embed_dim, label_num, **kwargs):
        super().__init__(**kwargs)
        
        self.ball = gt.PoincareBall(c=1.0)
        points = torch.randn(label_num, embed_dim) * 1e-5
        points = pmath.expmap0(points.to(cuda_device), k=self.ball.k)
        self.label_emb = gt.ManifoldParameter(points, manifold=self.ball)
        self.encoder = nn.Sequential(
            MobiusLinear(feature_num, embed_dim, bias=True, nonlin=None),
        )


    def regularization(self,points):
        return torch.norm(torch.norm(points, p=2, dim=1, keepdim=True) - 0.5, p=2, dim=1, keepdim=True)
    
    def radius_regularization(self,radius):
        return torch.norm(1-radius)

    def classifier(self,X):
        point_a = X.unsqueeze(1).expand(-1, self.label_emb.shape[0], -1) 
        point_b = self.label_emb.expand_as(point_a)
        logits = self.membership(point_a,point_b,dim=2).squeeze(2)
        return logits
    
    def forward(self, X,implication,exclusion):
        encoded = self.ball.projx(X)
        encoded = self.encoder(encoded)
        self.ball.assert_check_point_on_manifold(encoded)
        label_reg = self.regularization(self.label_emb)
        instance_reg = F.relu( torch.norm(encoded, p=2, dim=1, keepdim=True) - 0.95 ) + F.relu( 0.4 - torch.norm(encoded, p=2, dim=1, keepdim=True) )
        log_probability = self.classifier(encoded)
        # implication
        sub_label_id = implication[:,0]
        par_label_id = implication[:,1]
        sub_label_emb = self.label_emb[sub_label_id]
        par_label_emb = self.label_emb[par_label_id]
        inside_loss = F.relu(- self.insideness(sub_label_emb,par_label_emb))
        
        left_label_id = exclusion[:,0]
        right_label_id = exclusion[:,1]
        left_label_emb = self.label_emb[left_label_id]
        right_label_emb = self.label_emb[right_label_id]
        disjoint_loss = F.relu(- self.disjointedness(left_label_emb,right_label_emb) )
        return log_probability, inside_loss.mean(), disjoint_loss.mean(), label_reg.mean(), instance_reg.mean()
    
    def insideness(self, point_a, point_b,dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_a = (1 - point_a_dist**2 )/ (2*point_a_dist )
        radius_b = (1 - point_b_dist**2 )/ (2*point_b_dist )
        center_a = point_a*(1 + radius_a/point_a_dist)
        center_b = point_b*(1 + radius_b/point_b_dist)
        center_dist = torch.norm(center_a-center_b,p=2,dim=dim,keepdim=True)
        insideness =  (radius_b - radius_a) - center_dist
        return insideness
    
    def disjointedness(self, point_a, point_b,dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_a = (1 - point_a_dist**2 )/ (2*point_a_dist )
        radius_b = (1 - point_b_dist**2 )/ (2*point_b_dist )
        center_a = point_a*(1 + radius_a/point_a_dist)
        center_b = point_b*(1 + radius_b/point_b_dist)
        center_dist = torch.norm(center_a-center_b,p=2,dim=dim,keepdim=True)
        disjointedness = center_dist - (radius_a + radius_b)
        return disjointedness
    
    def membership(self, point_a, point_b,dim=-1):
        center_a = point_a
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_b = (1 - point_b_dist**2 )/ (2*point_b_dist )
        center_b = point_b*(1 + radius_b/point_b_dist)
        center_dist = torch.norm(center_a-center_b,p=2,dim=dim,keepdim=True)
        membership =  radius_b - center_dist
        return membership