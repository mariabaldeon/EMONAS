import numpy as np

#Eliminates the outliers computing the mean and std. dev in the whole dataset. Elimate all that are out of +/- stdv
def del_outall(X, stdv):
    X_new=np.copy(X)
    mean=np.mean(X)
    std_dv=np.std(X)
    pix_max=int(mean+stdv*std_dv)
    pix_min=int(mean-stdv*std_dv)
    k=0
    for i in range(0, X.shape[0]): 
        for j in range (0,X.shape[1]): 
            
            if X[i,j]< pix_min: 
                k+=1
                X_new[i,j]=pix_min
                
            elif X[i,j]>pix_max:
                X_new[i,j]=pix_max
                k+=1
    return X_new

#Eliminates the outliers computing the mean and std. dev for each image. Elimate all that are out of +/- stdv
def del_out(X, stdv): 
    X_new=np.copy(X)
    k=0
    for i in range(0, X.shape[0]): 
        Xi=X[i,:]
        mean=np.mean(Xi)
        std_dv=np.std(Xi)
        pix_max=int(mean+stdv*std_dv)
        pix_min=int(mean-stdv*std_dv)
        for j in range(0, Xi.shape[0]): 
            if Xi[j]< pix_min: 
                X_new[i,j]=pix_min
                k+=1
            
            elif Xi[j]>pix_max:
                X_new[i,j]=pix_max
                k+=1
    return X_new

#Eliminates the outliers computing the quartiles and interquartile for each image. Elimate all that are out of +/- qrt
def del_outiqr(X, qrt): 
    X_new=np.copy(X)
    k=0
    for i in range(0, X.shape[0]): 
        Xi=X[i,:]
        quartile_1, quartile_3 = np.percentile(Xi, [25, 75])
        iqr = quartile_3 - quartile_1
        pix_min = int(quartile_1 - (iqr * qrt))
        pix_max = int(quartile_3 + (iqr * qrt))
        for j in range(0, Xi.shape[0]): 
            if Xi[j]< pix_min: 
                X_new[i,j]=pix_min
                k+=1
            
            elif Xi[j]>pix_max:
                X_new[i,j]=pix_max
                k+=1
    return X_new

# Normalizes each slice by dividing by the maximum value
def norm_max(X): 
    X_new=np.copy(X)
    for i in range(0, X.shape[0]): 
        Xi=X[i,:]
        Xi=Xi/np.max(Xi)
        X_new[i,:]=Xi
    return X_new

# Normalizes each slice by dividing by subtracting the mean and diving by the std
def normalization_sl(X): 
    X_new=np.copy(X)
    for i in range(0, X.shape[0]): 
        Xi=X[i,:]
        Xi=(Xi-np.mean(Xi))/np.std(Xi)
        X_new[i,:]=Xi
    return X_new

#Normalizes the dataset by subtracting the whole mean and std deviation of the dataset
def normalization(dataset): 
    data_set=(data_set-np.mean(data_set))/np.std(data_set)
    return data_set,np.mean(data_set), np.std(data_set)

#Eliminates the outliers computing the mean and std. dev for each image. Elimate all that are out of +/- stdv
def del_out_3D(X, stdv): 
    X_new=np.copy(X)
    num, h, w, s, c=X.shape
    for i in range(num):
        for k in range(s):
            Xi=X[i,:,:,k,0]
            mean=np.mean(Xi)
            std_dv=np.std(Xi)
            pix_max=mean+stdv*std_dv
            pix_min=mean-stdv*std_dv
            for j in range(h):
                for m in range(w):
                    if Xi[j,m]< pix_min: 
                        X_new[i,j,m,k,0]=pix_min
                
                    elif Xi[j,m]>pix_max:
                        X_new[i,j,m,k,0]=pix_max
    return X_new

# Normalizes each slice by dividing by the maximum value
def norm_max_3D(X): 
    eps=1e-100
    X_new=np.copy(X)
    num, h, w, s, c=X.shape
    for i in range(num):
        for k in range(s):
            Xi=X[i,:,:,k,0]
            Xi=Xi-np.min(Xi)
            Xi=Xi/(np.max(Xi)+eps)
            X_new[i,:,:,k,0]=Xi
    return X_new

#Eliminates the outliers computing the mean and std. dev for each image. Elimate all that are out of +/- stdv
# Metrics are computed on the whole 3D image
def del_out_vol(X, stdv): 
    X_new=np.copy(X)
    num, h, w, s, c=X.shape
    #for each volume
    for i in range(num):
        print("image num", i)
        # take 3D image
        Xi=X[i,:,:,:,0]
        mean=np.mean(Xi)
        std_dv=np.std(Xi)
        print('mean ', mean)
        print("std_dv ", std_dv)
        pix_max=mean+stdv*std_dv
        pix_min=mean-stdv*std_dv
        # for each slice
        for k in range(s):
            #for each row
            for j in range(h):
                #for each column
                for m in range(w):
                    if Xi[j,m,k]< pix_min: 
                        X_new[i,j,m,k,0]=pix_min
                
                    elif Xi[j,m,k]>pix_max:
                        X_new[i,j,m,k,0]=pix_max
    return X_new

# Normalizes each volume by dividing by the maximum value
def norm_max_vol(X): 
    eps=1e-100
    X_new=np.copy(X)
    num, h, w, s, c=X.shape
    #for each image
    for i in range(num):
        Xi=X[i,:,:,:,0]
        Xi=Xi-np.min(Xi)
        Xi=Xi/(np.max(Xi)+eps)
        X_new[i,:,:,:,0]=Xi
    return X_new
