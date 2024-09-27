function unet()
    model = Chain(
        Conv((3,3),1=>64,pad=SamePad(),relu),
        Conv((3,3),64=>64,pad=SamePad(),relu),

        SkipConnection(
            Chain(
                MaxPool((2,2)),
                Conv((3,3),64=>128,pad=SamePad(),relu),
                Conv((3,3),128=>128,pad=SamePad(),relu),
                
                SkipConnection(
                    Chain(
                        MaxPool((2,2)),
                        Conv((3,3),128=>256,pad=SamePad(),relu),
                        Conv((3,3),256=>256,pad=SamePad(),relu),
                        
                        SkipConnection(
                            Chain(
                                MaxPool((2,2)),
                                Conv((3,3),256=>512,pad=SamePad(),relu),
                                Conv((3,3),512=>512,pad=SamePad(),relu),

                                SkipConnection(
                                    Chain(
                                        MaxPool((2,2)),
                                        Conv((3,3),512=>1024,pad=SamePad(),relu),
                                        Conv((3,3),1024=>1024,pad=SamePad(),relu),
                                        ConvTranspose((2,2),1024=>1024,pad=SamePad(),stride=2)
                                    ), 
                                    (mx,x)->cat(mx,x,dims=3)
                                ),

                                Conv((3,3),1536=>512,pad=SamePad(),relu),
                                Conv((3,3),512=>512,pad=SamePad(),relu),
                                ConvTranspose((2,2),512=>512,pad=SamePad(),stride=2)
                            ), 
                            (mx,x)->cat(mx,x,dims=3)
                        ),
                        
                        Conv((3,3),768=>256,pad=SamePad(),relu),
                        Conv((3,3),256=>256,pad=SamePad(),relu),
                        ConvTranspose((2,2),256=>256,pad=SamePad(),stride=2)
                    ), 
                    (mx,x)->cat(mx,x,dims=3)
                ),
                    
                Conv((3,3),384=>128,pad=SamePad(),relu),
                Conv((3,3),128=>128,pad=SamePad(),relu),
                ConvTranspose((2,2),128=>128,pad=SamePad(),stride=2)
            ),
            (mx,x)->cat(mx,x,dims=3)
        ),        
        
        Conv((3,3),192=>64,pad=SamePad(),relu),
        Conv((3,3),64=>64,pad=SamePad(),relu),
        Conv((3,3),64=>1,pad=SamePad(),relu)
    )
    
    return model
end
