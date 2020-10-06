# -*- encoding=utf8 -*-
import torch
import torch.nn as nn
import math

from module import BiaffineAttention,MultiHeadAttention,PositionwiseFeedForward



class TransformerEncoderLayer(nn.Module):

    def __init__(self , dim , heads , d_ff, dropout , windowsize,type="center",norm_after=False):
        super(TransformerEncoderLayer , self).__init__()

        self.self_atten = MultiHeadAttention(dim , heads , dropout , window=windowsize,type=type)
        self.feed_forward = PositionwiseFeedForward(dim , d_ff , dropout, norm_after)
        self.layer_norm1 = nn.LayerNorm(dim , eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.norm_after = norm_after


    def forward(self, inputs , mask):
        input_norm = self.norm_layers(inputs , self.layer_norm1 , False)
        context = self.self_atten(input_norm , input_norm , input_norm , mask=mask)
        out = self.dropout(context) + inputs
        out = self.norm_layers(out , self.layer_norm1 , True)

        out = self.feed_forward(out)
        return out
        
    def norm_layers(self, x,normlayer,after=True):
        if self.norm_after:
            if after:
                return normlayer(x)
            else:
                return x
        else:
            if after:
                return x
            else:
                return normlayer(x)

class BiaffineScoreLayer(nn.Module):
    def __init__(self,dim,classdim,dropout,segwords = True,midfeas=True,gate=False):
        super(BiaffineScoreLayer,self).__init__()
        self.gate = gate
        if gate:
            self.gate_f = nn.Sequential(nn.Linear(dim,dim),nn.Sigmoid())
            self.gate_b = nn.Sequential(nn.Linear(dim,dim),nn.Sigmoid())
        self.biaffineScore = BiaffineAttention(dim,classdim,dropout,True,True)
        if segwords:
            self.segwords = True
            self.linef= nn.Linear(dim,dim)
            self.linec=nn.Linear(dim,dim)
            self.lineb = nn.Linear(dim,dim)
        else:
            self.segwords = False

        if midfeas :
            self.midfeas = True
            self.embed = nn.Linear(6, dim)
            self.embed2 = nn.Linear(6, dim)
            self.embed3 = nn.Linear(6, dim)
        else:
            self.midfeas = False



    def forward(self, f,c,b,mask=None):
        if self.segwords == False and self.midfeas == False:
            return None,None,None,None,None

        maxlens = f.size(1)
        batchs = f.size(0)

        f2 = f
        c2 = c
        b2 = b
        if self.gate:
            gateb = self.gate_b(b+c)
            gatef = self.gate_f(f+c)
            f = f * gatef + c * (1-gatef)
            b = b * gateb + c * (1-gateb)
        else:
            f = f + c
            b = b + c
        f = f[:,:-1,:]
        b = b[:,1:,:]
        asw = self.biaffineScore(f,b)
        score = torch.argmax(asw, 2)

        if self.midfeas:
            addones = torch.zeros(batchs , 1 , 3)
            #addones[:,:,0]=1
            if torch.cuda.is_available():
               addones = addones.cuda()
            asw1 = torch.cat([asw,addones],-2)
            asw2 = torch.cat([addones , asw],-2)
            asw3 = torch.cat([asw1,asw2],-1)
            al1 = self.embed(asw3)
            al2 = self.embed3(asw3)
            al3 = self.embed2(asw3)
            if mask is not None:
                al1 = al1.masked_fill(mask.transpose(-1,-2) , 0 )
                al2 = al2.masked_fill(mask.transpose(-1,-2) , 0 )
                al3 = al3.masked_fill(mask.transpose(-1,-2) , 0 )
        else:
            al1 = None
            al2 = None
            al3 = None

        if self.segwords:
            adds = torch.ones(batchs , 1) * 2.0
            if torch.cuda.is_available():
                adds=adds.cuda()

            score = torch.cat([adds.long(),score] ,-1)
            score = score.data.eq(2)

            window = torch.ones(maxlens, maxlens, dtype=torch.uint8)

            window_masks = torch.tril(window).transpose(1,0)

            if torch.cuda.is_available():
                window_masks = window_masks.cuda() #to(torch.cuda.current_device())
            score = torch.matmul(score.float(),window_masks.float())

            score2 = score.unsqueeze(-1).expand(batchs,maxlens,maxlens).transpose(2,1)

            masks = torch.eq(score2 , score.unsqueeze(-1))

            if mask is not None:
                masks = masks.masked_fill(mask,0)
            masks = masks.transpose(-2,-1)
        else:
            masks = None

        return asw,masks,al1,al2,al3




class Addwords(nn.Module):
    def __init__(self, dropout,dim ,normafter,seg=True,mid=True ):
        super(Addwords,self).__init__()

        if mid:
            self.mid = mid
            self.dropout_f2 = nn.Dropout(dropout)
            self.dropout_c2 = nn.Dropout(dropout)
            self.dropout_b2 = nn.Dropout(dropout)
        else:
            self.mid = mid
     

        if seg:
            self.seg = seg
            self.dropout_f = nn.Dropout(dropout)
            self.dropout_c = nn.Dropout(dropout)
            self.dropout_b = nn.Dropout(dropout)
            self.line_b = nn.Linear(dim,dim)
            self.line_f = nn.Linear(dim,dim)
            self.line_c = nn.Linear(dim,dim)
        else:
            self.seg = seg

        if mid or seg:
            self.normafter = normafter
            self.normlayer_c = nn.LayerNorm(dim,eps=1e-6)
            self.normlayer_b = nn.LayerNorm(dim,eps=1e-6)
            self.normlayer_f = nn.LayerNorm(dim,eps=1e-6)


    def forward(self,f,c,b,mask , labels = None,label2=None,label3=None,masks=None):
        if self.seg == False and self.mid == False:
            return f,c,b
        b = self.normlayer(b,self.normlayer_b,True)
        c = self.normlayer(c,self.normlayer_c,True)
        f = self.normlayer(f,self.normlayer_f,True)


        if self.seg:
            b2 = self.dropout_b(self.line_b(torch.matmul(b.transpose(-2,-1).float(),mask.float()).transpose(-1,-2))) + b
            c2 = self.dropout_c(self.line_c(torch.matmul(c.transpose(-2,-1).float(),mask.float()).transpose(-1,-2))) + c
            f2 = self.dropout_f(self.line_f(torch.matmul(f.transpose(-2,-1).float(),mask.float()).transpose(-1,-2))) + f
            if masks is not None:
                b2 = b2.masked_fill(masks.transpose(-1,-2) , 0 )
                c2 = c2.masked_fill(masks.transpose(-1,-2) , 0 )
                f2 = f2.masked_fill(masks.transpose(-1,-2) , 0 )
        else:
            b2 = b
            c2 = c
            f2 = f	
        
        if self.mid and labels is not None:
            b2 = b2 + self.dropout_b2(labels)
            f2 = f2 + self.dropout_f2(label2)
            c2 = c2 + self.dropout_c2(label3)


        b = self.normlayer(b2 , self.normlayer_b,False)
        c = self.normlayer(c2 , self.normlayer_c,False)
        f = self.normlayer(f2 , self.normlayer_f,False)

        return f,c,b


    
    def normlayer(self,x,normlayer,layers):
        if self.normafter:
            if layers == False:
                return normlayer(x)
            else:
                return x
        else:
            if layers:
                return normlayer(x)
            else:
                return x




class TransformerEncoder(nn.Module):


    def __init__(self , dim , layers , heads , d_ff,dropout , embeddings,windowsize , norm_after,seglayer=False,seg=True,mid=True,gate=False):
        super(TransformerEncoder,self).__init__()
        self.dim=dim
        self.layers = layers
        self.heads = heads
        self.d_ff=d_ff
        self.dropout = nn.Dropout(dropout)
        self.embeddings = embeddings
        if norm_after == False:
            self.normforward = nn.LayerNorm(dim,eps=1e-6)
            self.normbackward = nn.LayerNorm(dim,eps=1e-6)
            self.norm = nn.LayerNorm(dim,eps=1e-6)
        self.norm_after = norm_after
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(dim , heads , d_ff,dropout,0,"center",norm_after) for j in range(layers)
            ]
        )

        self.transformerforward = nn.ModuleList(
            [
                TransformerEncoderLayer(dim, heads, d_ff, dropout,windowsize, "forward",norm_after) for j in range(layers)
            ]
        )

        self.transformerbackward = nn.ModuleList(
            [
                TransformerEncoderLayer(dim, heads, d_ff, dropout, windowsize, "backward",norm_after) for j in range(layers)
            ]
        )
        self.gate = gate
        if gate:
            self.gate_b = nn.Sequential(nn.Linear(dim,dim),nn.Sigmoid())
            self.gate_f = nn.Sequential(nn.Linear(dim,dim),nn.Sigmoid())

        self.seglayer = seglayer
        if seglayer:
            self.biaffineScores = BiaffineScoreLayer(dim,3,dropout, seg, mid,gate)
            self.addlayers = Addwords(dropout,dim,norm_after,seg,mid) 
            self.seg = seg
            self.mid = mid


    def forward(self, src ):
        embedds = self.embeddings(src)
        padding_idx=0
        src = src.unsqueeze(-1)
        words = src[:,:,0]
        mask = words.data.eq(padding_idx).unsqueeze(1)

        out1 = embedds
        out2 = embedds
        out3 = embedds
        for j in range(self.layers):
            if self.seglayer:
                if j == self.layers // 2:
                    score,masks,labels,label2,labels3 = self.biaffineScores(out2,out1,out3,mask)
                    out2,out1,out3 = self.addlayers(out2,out1,out3,masks,labels,label2,labels3,mask)
                    out1 = embedds + out1
                    out2 = embedds + out2
                    out3 = embedds + out3

            out2 = self.transformerforward[j](out2 , mask)
            out1 = self.transformer[j](out1, mask)
            out3 = self.transformerbackward[j](out3, mask)
        
        if self.norm_after == False:
            out2 = self.normforward(out2)
            out3 = self.normbackward(out3)
            out1 = self.norm(out1)
        if self.gate:
            gateb = self.gate_b(out3+out1)
            gatef = self.gate_f(out2+out1)
            out3 = out3 * gateb + out1 *(1-gateb)
            out2 = out2 * gatef + out1 *(1-gatef)
        else:
            out3 = out3+out1
            out2 = out2+out1
        if self.seglayer:
            return out2,out3,score
        return out2,out3,None


        

class BiaffineSegmentationModel(nn.Module):

    def __init__(self , dim , layers , heads , d_ff , dropout,embeddings,window,norm_after=True,seglayer=False,seg=True,mid=True,gate=False):
        super(BiaffineSegmentationModel , self).__init__()

        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.embeddings = embeddings
        self.d_ff = d_ff
        self.transformerEncoder = TransformerEncoder(dim,layers,heads,d_ff,dropout,embeddings,window,norm_after,seglayer,seg,mid,gate)
        self.biaffineScore = BiaffineAttention(dim ,3,dropout,True,True)


    def forward(self, input):
        outsforward,outbackward,score = self.transformerEncoder(input)

        h = outsforward[:,:-1,:]
        d = outbackward[:,1:,:]
        scoremodels = self.biaffineScore(h,d)
        return scoremodels,score


