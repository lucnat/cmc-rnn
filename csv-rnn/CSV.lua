
CSV = {}

CSV.read = function (path)
   -- Reads a csv file with comma delimiter and returns resulting tensor
   local csvFile = io.open(path, 'r')
   local firstLine = csvFile:read():split(',')
   local m = #firstLine
    
   -- loop through file in order to get file length
   csvFile:seek("set")        -- restore position
   local amountOfLines = 0
   for line in csvFile:lines('*l') do
      amountOfLines = amountOfLines + 1
   end
   csvFile:seek("set")        -- restore position
   local X = torch.zeros(amountOfLines,m)
   local i = 0
   for line in csvFile:lines('*l') do
      if i%400 == 0 then
        print('read another 400 lines')
      end
      i = i + 1
      local l = line:split(',')
      for key,val in ipairs(l) do
         X[i][key] = val
      end
   end
   return X
end

CSV.write = function (M,name)
   local out = assert(io.open("./"..name, "w")) -- open a file for serialization
   splitter = ","
   for i=1,M:size(1) do
       for j=1,M:size(2) do
           out:write(M[i][j])
           if j == M:size(2) then
               out:write("\n")
           else
               out:write(splitter)
           end
       end
   end
   out:close()
end

return CSV