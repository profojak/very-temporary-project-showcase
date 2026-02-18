#ifndef EPC_VECTOR_H
#define EPC_VECTOR_H

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace epc {

template <typename T, size_t N> class vector {
private:
  size_t size_;
  size_t capacity_;
  T *data_;
  alignas(T) std::byte buffer_[sizeof(T) * N];

  bool is_long() const { return capacity_ > N; }

  void deallocate_if_long() {
    if (is_long())
      ::operator delete[](data_, std::align_val_t(alignof(T)));
  }

public:
  vector() noexcept
      : size_(0u), capacity_(N), data_(reinterpret_cast<T *>(buffer_)) {}

  vector(const vector &other)
      : size_(0u), capacity_(N), data_(reinterpret_cast<T *>(buffer_)) {
    if (other.size_ > N) {
      data_ = static_cast<T *>(static_cast<void *>(::operator new[](
          other.size_ * sizeof(T), std::align_val_t(alignof(T)))));
      capacity_ = other.size_;
    }
    try {
      std::uninitialized_copy(other.data_, other.data_ + other.size_, data_);
      size_ = other.size_;
    } catch (...) {
      deallocate_if_long();
      throw;
    }
  }
  vector &operator=(const vector &other) {
    if (this != &other) {
      if (other.size_ <= capacity_) {
        if (other.size_ > size_) {
          std::copy(other.data_, other.data_ + size_, data_);
          std::uninitialized_copy(other.data_ + size_,
                                  other.data_ + other.size_, data_ + size_);
        } else {
          std::copy(other.data_, other.data_ + other.size_, data_);
          std::destroy(data_ + other.size_, data_ + size_);
        }
        size_ = other.size_;
      } else {
        T *new_data = static_cast<T *>(static_cast<void *>(::operator new[](
            other.size_ * sizeof(T), std::align_val_t(alignof(T)))));
        try {
          std::uninitialized_copy(other.data_, other.data_ + other.size_,
                                  new_data);
        } catch (...) {
          ::operator delete[](new_data, std::align_val_t(alignof(T)));
          throw;
        }
        std::destroy_n(data_, size_);
        deallocate_if_long();
        data_ = new_data;
        size_ = other.size_;
        capacity_ = other.size_;
      }
    }
    return *this;
  }

  vector(vector &&other) noexcept
      : size_(0u), capacity_(N), data_(reinterpret_cast<T *>(buffer_)) {
    if (other.is_long()) {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.data_ = reinterpret_cast<T *>(other.buffer_);
      other.size_ = 0u;
      other.capacity_ = N;
    } else {
      std::uninitialized_move(other.data_, other.data_ + other.size_, data_);
      std::destroy_n(other.data_, other.size_);
      size_ = other.size_;
      other.size_ = 0u;
    }
  }
  vector &operator=(vector &&other) noexcept {
    if (this != &other) {
      std::destroy_n(data_, size_);
      if (other.is_long()) {
        deallocate_if_long();
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = reinterpret_cast<T *>(other.buffer_);
        other.size_ = 0u;
        other.capacity_ = N;
      } else {
        if (is_long()) {
          deallocate_if_long();
          data_ = reinterpret_cast<T *>(buffer_);
          capacity_ = N;
        }
        std::uninitialized_move(other.data_, other.data_ + other.size_, data_);
        std::destroy_n(other.data_, other.size_);
        size_ = other.size_;
        other.size_ = 0u;
      }
    }
    return *this;
  }

  ~vector() {
    std::destroy_n(data_, size_);
    deallocate_if_long();
  }

  T *data() { return data_; }
  const T *data() const { return data_; }

  T &operator[](size_t i) { return data_[i]; }
  const T &operator[](size_t i) const { return data_[i]; }

  void push_back(const T &value) {
    if (size_ == capacity_)
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    std::construct_at(data_ + size_, value);
    size_++;
  }

  void push_back(T &&value) {
    if (size_ == capacity_)
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    std::construct_at(data_ + size_, std::move(value));
    size_++;
  }

  template <typename... Ts> void emplace_back(Ts &&...args) {
    if (size_ == capacity_)
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    std::construct_at(data_ + size_, std::forward<Ts>(args)...);
    size_++;
  }

  void pop_back() {
    std::destroy_at(data_ + size_ - 1u);
    size_--;
  }

  void clear() {
    std::destroy_n(data_, size_);
    size_ = 0u;
  }

  void reserve(size_t n) {
    if (n <= capacity_)
      return;
    T *new_data = static_cast<T *>(static_cast<void *>(
        ::operator new[](n * sizeof(T), std::align_val_t(alignof(T)))));
    try {
      std::uninitialized_move(data_, data_ + size_, new_data);
    } catch (...) {
      ::operator delete[](new_data, std::align_val_t(alignof(T)));
      throw;
    }
    std::destroy_n(data_, size_);
    deallocate_if_long();
    data_ = new_data;
    capacity_ = n;
  }

  size_t capacity() const { return capacity_; }
  size_t size() const { return size_; }

  void swap(vector &other) {
    if (is_long() && other.is_long()) {
      std::swap(data_, other.data_);
      std::swap(size_, other.size_);
      std::swap(capacity_, other.capacity_);
    }

    else if (!is_long() && !other.is_long()) {
      if (size_ < other.size_) {
        std::swap_ranges(data_, data_ + size_, other.data_);
        std::uninitialized_move(other.data_ + size_, other.data_ + other.size_,
                                data_ + size_);
        std::destroy(other.data_ + size_, other.data_ + other.size_);
      } else {
        std::swap_ranges(data_, data_ + other.size_, other.data_);
        std::uninitialized_move(data_ + other.size_, data_ + size_,
                                other.data_ + other.size_);
        std::destroy(data_ + other.size_, data_ + size_);
      }
      std::swap(size_, other.size_);
    }

    else {
      vector *long_vec = is_long() ? this : &other;
      vector *short_vec = is_long() ? &other : this;
      T *dest_buffer = reinterpret_cast<T *>(long_vec->buffer_);
      std::uninitialized_move(short_vec->data_,
                              short_vec->data_ + short_vec->size_, dest_buffer);
      std::destroy_n(short_vec->data_, short_vec->size_);
      T *long_vec_data = long_vec->data_;
      size_t long_vec_capacity = long_vec->capacity_;
      size_t long_vec_size = long_vec->size_;
      long_vec->data_ = dest_buffer;
      long_vec->size_ = short_vec->size_;
      long_vec->capacity_ = N;
      short_vec->data_ = long_vec_data;
      short_vec->size_ = long_vec_size;
      short_vec->capacity_ = long_vec_capacity;
    }
  }
};

} // namespace epc

#endif
