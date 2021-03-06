rotate = function(x) (apply(x, 2, rev))
cropBack = function(image,step,thresholdRatio,thresholdC){
  imageData = image %>%
    channel(mode = 'gray') %>%
    imageData()
  imageDataU = imageData
  imageDataD = rotate(imageData)
  imageDataL = t(imageData)
  imageDataR = rotate(imageDataL)
  iList = list(imageDataU, imageDataD, imageDataL, imageDataR)
  thresholdSize = min(dim(imageData))*thresholdRatio
  idx = lapply(iList, cropBoundary, step=step, thresholdSize = thresholdSize,thresholdC = thresholdC)
  idx = do.call(c,idx)
  xmin = idx[1]
  xmax = nrow(imageData)-idx[2]
  ymin = idx[3]
  ymax = ncol(imageData)-idx[4]
#   print(idx)
#   print(dim(imageData))
#   img = (imageData[idx[1]:(nrow(imageData)-idx[2]), idx[3]:(ncol(imageData)-idx[4])])
#   file = paste(paste("../../cropBack",name,sep = "/"), "jpg", sep = ".")
#   writeImage(img, file)
#   file0 = paste(paste("../../cropBack",name,sep = "/"), "0.jpg", sep = "")
#   writeImage(imageData,file0)
  return(c(xmin,xmax,ymin,ymax))
}