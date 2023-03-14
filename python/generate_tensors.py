def generate_tensors():
    (train_data_indices, test_data_indices) = utils.create_random_int_arrays(900, 100)
    train_data = []
    test_data = []
    for i in train_data_indices:
        id = str(i).zfill(6)
        # img = ImageEntity("data/face_synthetics/"+id+".png", "data/face_synthetics/"+id+"_ldmks.txt")
        img = Image.open("data/face_synthetics/"+id+".png")
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(img)
        torch.save(img_tensor, "data/face_synthetics/tensors_train_0/"+id+"_tensor.pt")
        print("data/face_synthetics/tensors_train_0/"+id+"_tensor.pt OK")

    for i in test_data_indices:
        id = str(i).zfill(6)
        # img = ImageEntity("data/face_synthetics/"+id+".png", "data/face_synthetics/"+id+"_ldmks.txt")
        img = Image.open("data/face_synthetics/"+id+".png")
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(img)
        torch.save(img_tensor, "data/face_synthetics/tensors_test_0/"+id+"_tensor.pt")
        print("data/face_synthetics/tensors_test_0/"+id+"_tensor.pt OK")


    # Writing indices
    file = open("data/face_synthetics/tensors_train_0/indices.txt","w")
    for indice in train_data_indices:
        file.write(str(indice) + " ")
    file.close()

    file = open("data/face_synthetics/tensors_test_0/indices.txt","w")
    for indice in test_data_indices:
        file.write(str(indice) + " ")
    file.close()
    # ###############