import numpy as np

def normalize(run_60=False, L2A=False, patch_size=128, train10=0, train20=0, train60=0, label=0, train=False):
    if patch_size == 32 and L2A == False:
        if run_60:
            train10[:,:,:,0]=(train10[:,:,:,0]-1302)/1491
            train10[:,:,:,1]=(train10[:,:,:,1]-1306)/1293
            train10[:,:,:,2]=(train10[:,:,:,2]-1400)/1348
            train10[:,:,:,3]=(train10[:,:,:,3]-2350)/1416

            train20[:,:,:,0]=(train20[:,:,:,0]-1505)/1441
            train20[:,:,:,1]=(train20[:,:,:,1]-2122)/1400
            train20[:,:,:,2]=(train20[:,:,:,2]-2428)/1447
            train20[:,:,:,3]=(train20[:,:,:,3]-2634)/1471
            train20[:,:,:,4]=(train20[:,:,:,4]-1985)/1199
            train20[:,:,:,5]=(train20[:,:,:,5]-1298)/1001
    
            train60[:,:,:,0]=(train20[:,:,:,0]-1627)/1297
            train60[:,:,:,1]=(train20[:,:,:,1]-821)/702
            
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1627)/1297
                label[:,:,:,1]=(label[:,:,:,1]-821)/702
                return train10, train20, train60, label
            else:
                return train10, train20, train60
            
        else:
            '''
            train10[:,:,:,0]=(train10[:,:,:,0]-1309)/1509
            train10[:,:,:,1]=(train10[:,:,:,1]-1311)/1307
            train10[:,:,:,2]=(train10[:,:,:,2]-1406)/1361
            train10[:,:,:,3]=(train10[:,:,:,3]-2357)/1439

            train20[:,:,:,0]=(train20[:,:,:,0]-1512)/1470
            train20[:,:,:,1]=(train20[:,:,:,1]-2128)/1434
            train20[:,:,:,2]=(train20[:,:,:,2]-2434)/1486
            train20[:,:,:,3]=(train20[:,:,:,3]-2640)/1514
            train20[:,:,:,4]=(train20[:,:,:,4]-1988)/1237
            train20[:,:,:,5]=(train20[:,:,:,5]-1301)/1032
            '''
            train10[:,:,:,0]=(train10[:,:,:,0]-1311)/1512
            train10[:,:,:,1]=(train10[:,:,:,1]-1313)/1309
            train10[:,:,:,2]=(train10[:,:,:,2]-1407)/1363
            train10[:,:,:,3]=(train10[:,:,:,3]-2355)/1443

            train20[:,:,:,0]=(train20[:,:,:,0]-1514)/1478
            train20[:,:,:,1]=(train20[:,:,:,1]-2127)/1446
            train20[:,:,:,2]=(train20[:,:,:,2]-2433)/1500
            train20[:,:,:,3]=(train20[:,:,:,3]-2638)/1530
            train20[:,:,:,4]=(train20[:,:,:,4]-1990)/1250
            train20[:,:,:,5]=(train20[:,:,:,5]-1303)/1042
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1512)/1470
                label[:,:,:,1]=(label[:,:,:,1]-2128)/1434
                label[:,:,:,2]=(label[:,:,:,2]-2434)/1486
                label[:,:,:,3]=(label[:,:,:,3]-2640)/1514
                label[:,:,:,4]=(label[:,:,:,4]-1988)/1237
                label[:,:,:,5]=(label[:,:,:,5]-1301)/1032
                #label[:,:,:,0]=(label[:,:,:,0]-1513)/1482
                #label[:,:,:,1]=(label[:,:,:,1]-2129)/1453
                #label[:,:,:,2]=(label[:,:,:,2]-2435)/1510
                #label[:,:,:,3]=(label[:,:,:,3]-2641)/1541
                #label[:,:,:,4]=(label[:,:,:,4]-1989)/1256
                #label[:,:,:,5]=(label[:,:,:,5]-1302)/1047
                return train10, train20, label
            else:
                return train10, train20
    
    elif patch_size == 32 and L2A == True:
        if run_60:
            train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1501
            return train10
        else:
            train10[:,:,:,0]=(train10[:,:,:,0]-1271)/1776
            train10[:,:,:,1]=(train10[:,:,:,1]-1171)/1676
            train10[:,:,:,2]=(train10[:,:,:,2]-911)/1671
            train10[:,:,:,3]=(train10[:,:,:,3]-2611)/1695

            train20[:,:,:,0]=(train20[:,:,:,0]-1614)/1749
            train20[:,:,:,1]=(train20[:,:,:,1]-2261)/1641
            train20[:,:,:,2]=(train20[:,:,:,2]-2509)/1634
            train20[:,:,:,3]=(train20[:,:,:,3]-2684)/1622
            train20[:,:,:,4]=(train20[:,:,:,4]-2192)/1377
            train20[:,:,:,5]=(train20[:,:,:,5]-1571)/1242
            
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1614)/1749
                label[:,:,:,1]=(label[:,:,:,1]-2261)/1641
                label[:,:,:,2]=(label[:,:,:,2]-2509)/1634
                label[:,:,:,3]=(label[:,:,:,3]-2684)/1622
                label[:,:,:,4]=(label[:,:,:,4]-2192)/1377
                label[:,:,:,5]=(label[:,:,:,5]-1571)/1242
                return train10, train20, label
            else:
                return train10, train20
    elif patch_size == 128 and L2A == False:
        if run_60:
            train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1493
            train10[:,:,:,1]=(train10[:,:,:,1]-1309)/1293
            train10[:,:,:,2]=(train10[:,:,:,2]-1403)/1349
            train10[:,:,:,3]=(train10[:,:,:,3]-2361)/1414

            train20[:,:,:,0]=(train20[:,:,:,0]-1511)/1443
            train20[:,:,:,1]=(train20[:,:,:,1]-2131)/1400
            train20[:,:,:,2]=(train20[:,:,:,2]-2439)/1446
            train20[:,:,:,3]=(train20[:,:,:,3]-2646)/1470
            train20[:,:,:,4]=(train20[:,:,:,4]-1994)/1200
            train20[:,:,:,5]=(train20[:,:,:,5]-1304)/1003
    
            train60[:,:,:,0]=(train20[:,:,:,0]-1629)/1300
            train60[:,:,:,1]=(train20[:,:,:,1]-825)/701
            
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1629)/1300
                label[:,:,:,1]=(label[:,:,:,1]-825)/701
                return train10, train20, train60, label
            else:
                return train10, train20, train60
                
        else:
            # 128 patches 20m 27000
            train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1501
            train10[:,:,:,1]=(train10[:,:,:,1]-1309)/1300
            train10[:,:,:,2]=(train10[:,:,:,2]-1403)/1354
            train10[:,:,:,3]=(train10[:,:,:,3]-2353)/1433

            train20[:,:,:,0]=(train20[:,:,:,0]-1509)/1462
            train20[:,:,:,1]=(train20[:,:,:,1]-2124)/1427
            train20[:,:,:,2]=(train20[:,:,:,2]-2430)/1480
            train20[:,:,:,3]=(train20[:,:,:,3]-2636)/1509
            train20[:,:,:,4]=(train20[:,:,:,4]-1991)/1236
            train20[:,:,:,5]=(train20[:,:,:,5]-1304)/1031
            
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1509)/1462
                label[:,:,:,1]=(label[:,:,:,1]-2124)/1427
                label[:,:,:,2]=(label[:,:,:,2]-2430)/1480
                label[:,:,:,3]=(label[:,:,:,3]-2636)/1509
                label[:,:,:,4]=(label[:,:,:,4]-1991)/1236
                label[:,:,:,5]=(label[:,:,:,5]-1304)/1031
                return train10, train60, label
            else:
                return train10, train20
                
    elif patch_size == 128 and L2A == True:
        if run_60:
            train10[:,:,:,0]=(train10[:,:,:,0]-1306)/1501
            return train10
        else:
            # 128 patches 20m 27000
            train10[:,:,:,0]=(train10[:,:,:,0]-1273)/1777
            train10[:,:,:,1]=(train10[:,:,:,1]-1171)/1677
            train10[:,:,:,2]=(train10[:,:,:,2]-911)/1672
            train10[:,:,:,3]=(train10[:,:,:,3]-2607)/1694

            train20[:,:,:,0]=(train20[:,:,:,0]-1615)/1749
            train20[:,:,:,1]=(train20[:,:,:,1]-2259)/1641
            train20[:,:,:,2]=(train20[:,:,:,2]-2505)/1633
            train20[:,:,:,3]=(train20[:,:,:,3]-2680)/1621
            train20[:,:,:,4]=(train20[:,:,:,4]-2192)/1376
            train20[:,:,:,5]=(train20[:,:,:,5]-1571)/1240
            
            if train == True:
                label[:,:,:,0]=(label[:,:,:,0]-1615)/1749
                label[:,:,:,1]=(label[:,:,:,1]-2259)/1641
                label[:,:,:,2]=(label[:,:,:,2]-2505)/1633
                label[:,:,:,3]=(label[:,:,:,3]-2680)/1621
                label[:,:,:,4]=(label[:,:,:,4]-2192)/1376
                label[:,:,:,5]=(label[:,:,:,5]-1571)/1240
                return train10, train20, label
            else:
                return train10, train20
                
def reverse_normalize(image, run_60=False, L2A=False, patch_size=32):
    if run_60 == True:
        image[:,:,0] = image[:,:,0] * 1297 + 1627
        image[:,:,1] = image[:,:,1] * 702 + 821
        return image
    elif L2A == False and patch_size==128:
        image[:,:,0] = image[:,:,0] * 1462 + 1509
        image[:,:,1] = image[:,:,1] * 1427 + 2124 
        image[:,:,2] = image[:,:,2] * 1480 + 2430
        image[:,:,3] = image[:,:,3] * 1509 + 2636
        image[:,:,4] = image[:,:,4] * 1236 + 1991
        image[:,:,5] = image[:,:,5] * 1031 + 1304
        return image
    elif L2A == False and patch_size==32:
        #image[:,:,0] = image[:,:,0] * 1470 + 1512
        #image[:,:,1] = image[:,:,1] * 1434 + 2128
        #image[:,:,2] = image[:,:,2] * 1486 + 2434
        #image[:,:,3] = image[:,:,3] * 1514 + 2640
        #image[:,:,4] = image[:,:,4] * 1237 + 1988
        #image[:,:,5] = image[:,:,5] * 1032 + 1301       
        image[:,:,0] = image[:,:,0] * 1478 + 1514
        image[:,:,1] = image[:,:,1] * 1446 + 2127
        image[:,:,2] = image[:,:,2] * 1500 + 2433
        image[:,:,3] = image[:,:,3] * 1530 + 2638
        image[:,:,4] = image[:,:,4] * 1250 + 1990
        image[:,:,5] = image[:,:,5] * 1042 + 1303
        #image[:,:,0] = image[:,:,0] * 1482 + 1513
        #image[:,:,1] = image[:,:,1] * 1453 + 2129
        #image[:,:,2] = image[:,:,2] * 1510 + 2435
        #image[:,:,3] = image[:,:,3] * 1541 + 2641
        #image[:,:,4] = image[:,:,4] * 1256 + 1989
        #image[:,:,5] = image[:,:,5] * 1047 + 1302
        return image
    else:
        image[:,:,0] = image[:,:,0] * 1749 + 1614
        image[:,:,1] = image[:,:,1] * 1641 + 2261
        image[:,:,2] = image[:,:,2] * 1634 + 2509
        image[:,:,3] = image[:,:,3] * 1622 + 2684
        image[:,:,4] = image[:,:,4] * 1377 + 2192
        image[:,:,5] = image[:,:,5] * 1242 + 1571      
        return image