
import numpy as np
from utils import build_poly

def remove_features(X, feature_names, jet_num):
    """ Remove the undefined features for a given jet """
    if jet_num == 0:
        X_ext = np.delete(X,[4,5,6,12,23,24,25,26,27,28,29], axis = 1).reshape((X.shape[0],X.shape[1]-11))
        new_feature_names = np.delete(feature_names, [4,5,6,12,23,24,25,26,27,28,29])
        equivalent = {7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16, 21: 17, 22: 18}
    elif jet_num == 1:
        X_ext = np.delete(X,[4,5,6,12,26,27,28], axis = 1).reshape((X.shape[0],X.shape[1]-7))
        new_feature_names = np.delete(feature_names, [4,5,6,12,26,27,28])
        equivalent = {7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 29: 22}
    else:
        X_ext = X
        new_feature_names = feature_names
        equivalent = None
    return X_ext, new_feature_names, equivalent

def extend_features(X, feature_names, degree, jet_num, add_arcsinh_feature = True, add_log_features = True, add_momentum_features = True, remove_jet_column = True, equivalent = None):
    """ Extend the features """
    # Remove pri_jet_num (if false , we consider that is has already been removed)
    correction = -1
    if remove_jet_column:
        if equivalent is not None:
            X_ext = np.delete(X, equivalent[22], axis = 1)
            feature_names =  np.delete(feature_names, equivalent[22])
        else:
            X_ext = np.delete(X, 22, axis = 1)
            feature_names = np.delete(feature_names, 22)
    else:
        X_ext = X

    if degree>1:
        # build polynomial basis function on the existsing data
        X_ext = build_poly(X_ext, degree)

    # adding degree names
    new_features_names = ['constant_feature']
    for i in range(len(feature_names)):
        for d in range(degree):
            new_features_names.append(feature_names[i] + '_power_' + str(d+1))
    # adding 1 feature
    if add_arcsinh_feature and jet_num > 1:
        #apply arcsinh on one feature tolinearize it
        feature = X[:, 6]
        feature = np.arcsinh(feature)
        new_features_names.append('arcsinh(DER_prodeta_jet_jet)')
        X_ext = np.concatenate((X_ext, feature.reshape((X.shape[0], 1))), axis = 1)

    # adding 7 (or 6) features
    if add_log_features :
        #apply log on 7 (or 6) features to become more linear
        if jet_num > 1:
            features = X[:, [0,1,2,3,5,9,10]]
            f = ['log(DER_mass_MMC)', 'log(DER_mass_transverse_met_lep)', 'log(DER_mass_vis)', 'log(DER_pt_h)', 'log(DER_mass_jet_jet)', 'log(DER_sum_pt)', 'log(DER_pt_ratio_lep_tau)']
        else:
            features = X[:, [0,1,2,3,equivalent[9],equivalent[10]]]
            f = ['log(DER_mass_MMC)', 'log(DER_mass_transverse_met_lep)', 'log(DER_mass_vis)', 'log(DER_pt_h)', 'log(DER_sum_pt)', 'log(DER_pt_ratio_lep_tau)']
        features = np.log(1+np.abs(features))

        for i in range(len(f)):
            # add new feature names
            new_features_names.append(f[i])

        X_ext = np.concatenate((X_ext, features), axis = 1)

    # adding 4 (or 2) features
    if add_momentum_features:
        if jet_num > 1:
            tra_mom = X[:,[13, 16, 23+correction, 26+correction]]
            eta_ang = X[:,[14, 17, 24+correction, 27+correction]]
            f = ['DER_tau_mom', 'DER_lep_mom', 'DER_jet_leading_mom', 'DER_jet_subleading_mom']
        else:
            tra_mom = X[:,[equivalent[13], equivalent[16]]]
            eta_ang = X[:,[equivalent[14], equivalent[17]]]
            f = ['DER_tau_mom', 'DER_lep_mom']

        #compute new features
        P_mom = tra_mom / np.sin(np.arctan(2*np.exp(- eta_ang)))

        for i in range(len(f)):
            # add new feature names
            new_features_names.append(f[i])

        X_ext = np.concatenate((X_ext, P_mom), axis = 1)

    return X_ext, new_features_names
