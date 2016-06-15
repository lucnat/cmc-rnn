
CSV = {}

CSV.read = function (path)
   -- Reads a csv file with comma delimiter and returns resulting tensor
   local csvFile = io.open(path, 'r')
   local firstLine = csvFile:read():split(',')
   local m = table.getn(firstLine)
   local N = csvFile:seek("end")/(2*m)
   csvFile:seek("set")        -- restore position

   local X = torch.zeros(N,m)
   local i = 0
   for line in csvFile:lines('*l') do
      i = i + 1
      local l = line:split(',')
      for key,val in ipairs(l) do
         X[i][key] = val
      end
   end
   return X
end

CSV.write = function (M)
   local out = assert(io.open("./samples.csv", "w")) -- open a file for serialization
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