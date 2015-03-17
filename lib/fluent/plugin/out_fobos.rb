module Fluent
  class FobosOutput < BufferedOutput
    Fluent::Plugin.register_output('fobos', self)

    def configure(conf)
      super
    end

    def start
      @t = 0         # count of data
      @c = 0.1       # training rate
      @lambda = 0.01 # regularize parameter
      @w = {}
      super
    end

    def shutdown
      super
    end

    def format(tag, time, record)
      [tag, time, record].to_msgpack
    end

    def write(chunk)
      chunk.msgpack_each {|tag,time,record|
        type = record['type']
        x    = record['x']
        # train
        case type
        when 'train'
          y = record['y'].to_f
          train(x,y)
        else
          puts "other"
        end
        $stderr.puts [tag,time,record].to_json
      }
    end

    private

    def train(x, y)
      @t += 1.0
      eta = @c/Math.sqrt(@t)
      # optimize loss function
      if get_loss(x, y) > 0.0
        x.each {|key, value|
          loss_grad = -1.0*y*value.to_f # gradient of loss function
          @w[key] = 0.0 unless @w.has_key?(key)
          @w[key] -= eta*loss_grad
        }
      end

      # optimize regularize function
      x.each {|key, value|
        next 0.0 unless @w.has_key?(key)
        if @w[key].abs < eta*@lambda
          @w.delete(key)
        else
          regularize_grad = sign(@w[key])*@lambda # gradient of regularize function
          @w[key] -= eta*regularize_grad
        end
      }

      $stderr.puts @w
    end

    def get_loss(x, y)
      # hinge loss
      p = 0.0 # inner product
      x.each {|key, value|
        w = @w.has_key?(key) ? @w[key] : 0.0
        p += w*value.to_f
      }
      [0.0, 1.0-y*p].max
    end

    def sign(n)
      return -1 if n < 0
      return  1 if n > 0
      0
    end
  end
end
