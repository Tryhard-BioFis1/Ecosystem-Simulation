from en_main import main

filename = input("Name of txt where the data will be stored : ")
iphyto = int(input("Number of PHYTO initial individuals : "))
iphago = int(input("Number of PHAGO initial individuals : "))
for _ in range(5):
    main(indv_phyto=iphyto, indv_phago=iphago, safe_data_at_file=filename)
